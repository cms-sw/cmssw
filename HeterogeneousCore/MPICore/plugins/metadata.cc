#include "metadata.h"

#include <cstring>
#include <iostream>

ProductMetadataBuilder::ProductMetadataBuilder() : buffer_(nullptr), capacity_(0), size_(0), readOffset_(0) {
  // reserve at least 9 bytes for header
  reserve(9);
  size_ = 9;
}

ProductMetadataBuilder::ProductMetadataBuilder(size_t expectedSize)
    : buffer_(nullptr), capacity_(0), size_(0), readOffset_(0) {
  reserve(expectedSize + 9);
  size_ = 9;
}

ProductMetadataBuilder::~ProductMetadataBuilder() { std::free(buffer_); }

ProductMetadataBuilder::ProductMetadataBuilder(ProductMetadataBuilder&& other) noexcept
    : buffer_(other.buffer_), capacity_(other.capacity_), size_(other.size_), readOffset_(other.readOffset_) {
  other.buffer_ = nullptr;
  other.capacity_ = 0;
  other.size_ = 0;
  other.readOffset_ = 0;
}

ProductMetadataBuilder& ProductMetadataBuilder::operator=(ProductMetadataBuilder&& other) noexcept {
  if (this != &other) {
    std::free(buffer_);
    buffer_ = other.buffer_;
    capacity_ = other.capacity_;
    size_ = other.size_;
    readOffset_ = other.readOffset_;
    other.buffer_ = nullptr;
    other.capacity_ = 0;
    other.size_ = 0;
    other.readOffset_ = 0;
  }
  return *this;
}

void ProductMetadataBuilder::reserve(size_t bytes) {
  if (capacity_ >= bytes)
    return;
  resizeBuffer(bytes);
}

void ProductMetadataBuilder::setHeader() {
  assert(size_ >= 9 && "Buffer must reserve space for header");
  std::memcpy(buffer_, &productCount_, sizeof(uint64_t));  // first 8 bytes
  buffer_[8] = productFlags_;                              // 9th byte
}

void ProductMetadataBuilder::addMissing() {
  productFlags_ |= HasMissing;
  append<uint8_t>(static_cast<uint8_t>(ProductMetadata::Kind::Missing));
}

void ProductMetadataBuilder::addSerialized(size_t size) {
  productFlags_ |= HasSerialized;
  append<uint8_t>(static_cast<uint8_t>(ProductMetadata::Kind::Serialized));
  append<size_t>(size);
}

void ProductMetadataBuilder::addTrivialCopy(const std::byte* buffer, size_t size) {
  productFlags_ |= HasTrivialCopy;
  append<uint8_t>(static_cast<uint8_t>(ProductMetadata::Kind::TrivialCopy));
  append<size_t>(size);
  appendBytes(buffer, size);
}

const uint8_t* ProductMetadataBuilder::data() const { return buffer_; }
uint8_t* ProductMetadataBuilder::data() { return buffer_; }
size_t ProductMetadataBuilder::size() const { return size_; }
std::span<const uint8_t> ProductMetadataBuilder::buffer() const { return {buffer_, size_}; }

void ProductMetadataBuilder::receiveMetadata(MPI_Message message, size_t size) {
  assert(size_ == 9 && "metadata receive buffer must be empty");
  assert(size != 0 && "metadata message is empty");
  resizeBuffer(static_cast<size_t>(size));
  MPI_Mrecv(buffer_, size, MPI_BYTE, &message, MPI_STATUS_IGNORE);
  if (size < 9)
    throw std::runtime_error("Metadata message too short");
  productCount_ = consume<size_t>();
  assert(productCount_ > 0 && "no products sent or product number not set");
  productFlags_ = consume<ProductFlags>();
  size_ += size;
}

ProductMetadata ProductMetadataBuilder::getNext() {
  if (readOffset_ >= size_)
    throw std::out_of_range("No more metadata entries");

  ProductMetadata meta;
  auto kind = static_cast<ProductMetadata::Kind>(consume<uint8_t>());
  meta.kind = kind;

  switch (kind) {
    case ProductMetadata::Kind::Missing:
      break;

    case ProductMetadata::Kind::Serialized:
      meta.sizeMeta = consume<size_t>();
      break;

    case ProductMetadata::Kind::TrivialCopy: {
      size_t blobSize = consume<size_t>();
      if (readOffset_ + blobSize > size_) {
        throw std::runtime_error("Metadata buffer too short for trivialCopy data");
      }
      meta.sizeMeta = blobSize;
      meta.trivialCopyOffset = buffer_ + readOffset_;
      readOffset_ += blobSize;
      break;
    }

    default:
      throw std::runtime_error("Unknown metadata kind");
  }

  return meta;
}

void ProductMetadataBuilder::resizeBuffer(size_t newCap) {
  uint8_t* newBuf = static_cast<uint8_t*>(std::realloc(buffer_, newCap));
  if (!newBuf)
    throw std::bad_alloc();
  buffer_ = newBuf;
  capacity_ = newCap;
}

void ProductMetadataBuilder::ensureCapacity(size_t needed) {
  if (size_ + needed <= capacity_)
    return;

  size_t newCapacity = capacity_ ? capacity_ : 64;
  while (size_ + needed > newCapacity)
    newCapacity *= 2;

  uint8_t* newData = static_cast<uint8_t*>(std::realloc(buffer_, newCapacity));
  if (!newData)
    throw std::bad_alloc();
  buffer_ = newData;
  capacity_ = newCapacity;
}

void ProductMetadataBuilder::appendBytes(const std::byte* src, size_t size) {
  ensureCapacity(size);
  std::memcpy(buffer_ + size_, src, size);
  size_ += size;
}

void ProductMetadataBuilder::debugPrintMetadataSummary() const {
  size_t offset = 0;
  size_t count = 0;
  size_t numMissing = 0;
  size_t numSerialized = 0;
  size_t numTrivial = 0;

  std::cerr << "---- ProductMetadata Debug Summary ----\n";
  while (offset < size_) {
    uint8_t kindVal = buffer_[offset];
    auto kind = static_cast<ProductMetadata::Kind>(kindVal);
    offset += sizeof(uint8_t);
    count++;

    std::cerr << "Product #" << count << ": ";

    switch (kind) {
      case ProductMetadata::Kind::Missing:
        numMissing++;
        std::cerr << "Missing\n";
        break;

      case ProductMetadata::Kind::Serialized: {
        if (offset + sizeof(size_t) > size_) {
          std::cerr << "ERROR: corrupted serialized metadata\n";
          return;
        }
        size_t sz;
        std::memcpy(&sz, buffer_ + offset, sizeof(size_t));
        offset += sizeof(size_t);
        numSerialized++;
        std::cerr << "Serialized, size = " << sz << "\n";
        break;
      }

      case ProductMetadata::Kind::TrivialCopy: {
        if (offset + sizeof(size_t) > size_) {
          std::cerr << "ERROR: corrupted trivial copy metadata\n";
          return;
        }
        size_t sz;
        std::memcpy(&sz, buffer_ + offset, sizeof(size_t));
        offset += sizeof(size_t);
        if (offset + sz > size_) {
          std::cerr << "ERROR: trivial copy data overflows buffer\n";
          return;
        }
        offset += sz;
        numTrivial++;
        std::cerr << "TrivialCopy, size = " << sz << "\n";
        break;
      }

      default:
        std::cerr << "Unknown kind: " << static_cast<int>(kindVal) << "\n";
        return;
    }
  }

  std::cerr << "----------------------------------------\n";
  std::cerr << "Total entries:   " << count << "\n";
  std::cerr << "  Missing:       " << numMissing << "\n";
  std::cerr << "  Serialized:    " << numSerialized << "\n";
  std::cerr << "  TrivialCopy:   " << numTrivial << "\n";
  std::cerr << "Total buffer size: " << size_ << " bytes\n";
}

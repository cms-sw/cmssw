// C++ standard library headers
#include <cstring>
#include <iostream>
#include <sstream>

// CMSSW headers
#include "HeterogeneousCore/MPICore/interface/metadata.h"

ProductMetadataBuilder::ProductMetadataBuilder() : buffer_(nullptr), capacity_(0), size_(0), readOffset_(0) {
  // reserve at least 13 bytes for header
  reserve(maxMetadataSize_);
  size_ = headerSize_;
}

ProductMetadataBuilder::ProductMetadataBuilder(int16_t productCount)
    : buffer_(nullptr), capacity_(0), size_(0), readOffset_(0) {
  // we need 1 byte for type, 8 bytes for size, and at least 8 bytes for MemoryCopyTraits Properties buffer
  // on average 24 bytes per product should be enough, but metadata will adapt if the actual contents is larger
  reserve(productCount * 24 + headerSize_);
  header_.productCount = static_cast<int16_t>(productCount);
  size_ = headerSize_;
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
  assert(size_ >= headerSize_ && "Buffer must reserve space for header");
  std::memcpy(buffer_, &header_, sizeof(header_));
}

void ProductMetadataBuilder::addMissing() {
  header_.productFlags |= HasMissing;
  append<uint8_t>(static_cast<uint8_t>(ProductMetadata::Kind::Missing));
}

void ProductMetadataBuilder::addSerialized(uint64_t size) {
  header_.productFlags |= HasSerialized;
  append<uint8_t>(static_cast<uint8_t>(ProductMetadata::Kind::Serialized));
  append<uint64_t>(size);
  header_.serializedBufferSize += static_cast<int32_t>(size);
}

void ProductMetadataBuilder::addTrivialCopy(const std::byte* buffer, uint64_t size) {
  header_.productFlags |= HasTrivialCopy;
  append<uint8_t>(static_cast<uint8_t>(ProductMetadata::Kind::TrivialCopy));
  append<uint64_t>(size);
  appendBytes(buffer, size);
}

const uint8_t* ProductMetadataBuilder::data() const { return buffer_; }
uint8_t* ProductMetadataBuilder::data() { return buffer_; }
size_t ProductMetadataBuilder::size() const { return size_; }
std::span<const uint8_t> ProductMetadataBuilder::buffer() const { return {buffer_, size_}; }

void ProductMetadataBuilder::receiveMetadata(int src, int tag, MPI_Comm comm) {
  MPI_Status status;
  MPI_Recv(buffer_, maxMetadataSize_, MPI_BYTE, src, tag, comm, &status);
  // add error handling if message is too long (quite unlikely to happen so far)
  int receivedBytes = 0;
  MPI_Get_count(&status, MPI_BYTE, &receivedBytes);
  assert(static_cast<size_t>(receivedBytes) >= headerSize_ && "received metadata was less than header size");
  memcpy(&header_, buffer_, sizeof(header_));
  size_ = receivedBytes;
  readOffset_ = headerSize_;
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
      meta.sizeMeta = consume<uint64_t>();
      break;

    case ProductMetadata::Kind::TrivialCopy: {
      uint64_t blobSize = consume<uint64_t>();
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
  if (size_ < sizeof(MetadataHeader)) {
    std::cerr << "ERROR: Buffer too small to contain metadata header\n";
    return;
  }

  std::ostringstream out;
  size_t offset = 0;

  // --- Read header fields explicitly ---
  MetadataHeader header;
  std::memcpy(&header, buffer_, sizeof(MetadataHeader));
  offset += sizeof(MetadataHeader);

  out << "---- ProductMetadata Debug Summary ----\n";
  out << "Header:\n";
  out << "  Product count:           " << header.productCount << "\n";
  out << "  Serialized buffer size:  " << header.serializedBufferSize << " bytes\n";
  out << "  Flags:";
  if (header.productFlags & HasMissing)
    out << " Missing";
  if (header.productFlags & HasSerialized)
    out << " Serialized";
  if (header.productFlags & HasTrivialCopy)
    out << " TrivialCopy";
  out << "\n\n";

  // --- Parse product entries ---
  size_t count = 0;
  size_t numMissing = 0;
  size_t numSerialized = 0;
  size_t numTrivial = 0;

  while (offset < size_) {
    if (count >= static_cast<size_t>(header.productCount)) {
      out << "WARNING: More metadata entries than header.productCount\n";
      break;
    }

    if (offset + sizeof(uint8_t) > size_) {
      out << "ERROR: Truncated metadata entry\n";
      break;
    }

    auto kind = static_cast<ProductMetadata::Kind>(buffer_[offset]);
    offset += sizeof(uint8_t);
    count++;

    out << "Product #" << count << ": ";

    switch (kind) {
      case ProductMetadata::Kind::Missing:
        numMissing++;
        out << "Missing\n";
        break;

      case ProductMetadata::Kind::Serialized: {
        if (offset + sizeof(size_t) > size_) {
          out << "ERROR: Corrupted serialized metadata\n";
          return;
        }
        size_t sz = 0;
        std::memcpy(&sz, buffer_ + offset, sizeof(size_t));
        offset += sizeof(size_t);
        numSerialized++;
        out << "Serialized, size = " << sz << "\n";
        break;
      }

      case ProductMetadata::Kind::TrivialCopy: {
        if (offset + sizeof(size_t) > size_) {
          out << "ERROR: Corrupted trivial-copy metadata\n";
          return;
        }
        size_t sz = 0;
        std::memcpy(&sz, buffer_ + offset, sizeof(size_t));
        offset += sizeof(size_t);

        if (offset + sz > size_) {
          out << "ERROR: Trivial-copy data overflows buffer\n";
          return;
        }
        offset += sz;
        numTrivial++;
        out << "TrivialCopy, size = " << sz << "\n";
        break;
      }

      default:
        out << "ERROR: Unknown product kind " << static_cast<int>(kind) << "\n";
        return;
    }
  }

  if (count != static_cast<size_t>(header.productCount)) {
    out << "\nWARNING: Parsed " << count << " entries, but header says " << header.productCount << "\n";
  }

  out << "\n----------------------------------------\n";
  out << "Total entries parsed:   " << count << "\n";
  out << "  Missing:              " << numMissing << "\n";
  out << "  Serialized:           " << numSerialized << "\n";
  out << "  TrivialCopy:          " << numTrivial << "\n";
  out << "Total metadata size:    " << size_ << " bytes\n";

  std::cerr << out.str() << std::flush;
}

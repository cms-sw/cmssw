// C++ standard library headers
#include <cstring>
#include <iostream>
#include <sstream>

// CMSSW headers
#include "HeterogeneousCore/MPICore/interface/metadata.h"

ProductMetadataBuilder::ProductMetadataBuilder() : buffer_(nullptr), capacity_(0), size_(0), readOffset_(0) {
  reserve(maxMetadataSize_);
  size_ = sizeof(MetadataHeader);
  header().productCount = 0;
  header().productFlags = 0;
  header().serializedBufferSize = 0;
}

ProductMetadataBuilder::ProductMetadataBuilder(int16_t productCount)
    : buffer_(nullptr), capacity_(0), size_(0), readOffset_(0) {
  // we need 1 byte for type, 8 bytes for size, and at least 8 bytes for MemoryCopyTraits Properties buffer
  // on average 24 bytes per product should be enough, but metadata will adapt if the actual contents is larger
  reserve(productCount * 24 + sizeof(MetadataHeader));
  size_ = sizeof(MetadataHeader);
  header().productCount = static_cast<int16_t>(productCount);
  header().productFlags = 0;
  header().serializedBufferSize = 0;
}

ProductMetadataBuilder::~ProductMetadataBuilder() {
  // Free the memory buffer.
  std::free(buffer_);
}

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

MetadataHeader& ProductMetadataBuilder::header() {
  assert(size_ >= sizeof(MetadataHeader) && "Invalid ProductMetadataBuilder, cannot access the MetadataHeader.");
  return *reinterpret_cast<MetadataHeader*>(buffer_);
}

MetadataHeader const& ProductMetadataBuilder::header() const {
  assert(size_ >= sizeof(MetadataHeader) && "Invalid ProductMetadataBuilder, cannot access the MetadataHeader.");
  return *reinterpret_cast<const MetadataHeader*>(buffer_);
}

void ProductMetadataBuilder::reserve(size_t bytes) {
  // do not shrink the buffer
  if (capacity_ >= bytes)
    return;

  // double the capacity until its enough for the requested size
  size_t newCapacity = capacity_ > 0 ? capacity_ : 64;
  while (bytes > newCapacity)
    newCapacity *= 2;

  // reallocate the buffer as needed, and move the contents to the new memory
  uint8_t* newBuffer = static_cast<uint8_t*>(std::realloc(buffer_, newCapacity));
  if (newBuffer == nullptr)
    throw std::bad_alloc();

  // if the reallocation succeded, update the data members
  buffer_ = newBuffer;
  capacity_ = newCapacity;
}

void ProductMetadataBuilder::addMissing() {
  header().productFlags |= HasMissing;
  append<uint8_t>(static_cast<uint8_t>(ProductMetadata::Kind::Missing));
}

void ProductMetadataBuilder::addSerialized(uint64_t size) {
  header().productFlags |= HasSerialized;
  append<uint8_t>(static_cast<uint8_t>(ProductMetadata::Kind::Serialized));
  append<uint64_t>(size);
  header().serializedBufferSize += static_cast<int32_t>(size);
}

void ProductMetadataBuilder::addTrivialCopy(const std::byte* buffer, uint64_t size) {
  header().productFlags |= HasTrivialCopy;
  append<uint8_t>(static_cast<uint8_t>(ProductMetadata::Kind::TrivialCopy));
  append<uint64_t>(size);
  appendBytes(buffer, size);
}

void ProductMetadataBuilder::receiveMetadata(int src, int tag, MPI_Comm comm) {
  MPI_Status status;
  MPI_Recv(buffer_, maxMetadataSize_, MPI_BYTE, src, tag, comm, &status);
  // add error handling if message is too long (quite unlikely to happen so far)
  int receivedBytes = 0;
  MPI_Get_count(&status, MPI_BYTE, &receivedBytes);
  assert(static_cast<size_t>(receivedBytes) >= sizeof(MetadataHeader) && "received metadata was less than header size");
  size_ = receivedBytes;
  readOffset_ = sizeof(MetadataHeader);
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

void ProductMetadataBuilder::appendBytes(const std::byte* src, size_t size) {
  reserve(size_ + size);
  std::memcpy(buffer_ + size_, src, size);
  size_ += size;
}

void ProductMetadataBuilder::consumeBytes(std::byte* dst, size_t size) {
  if (readOffset_ + size > size_)
    throw std::runtime_error("Buffer underflow");
  std::memcpy(dst, buffer_ + readOffset_, size);
  readOffset_ += size;
}

void ProductMetadataBuilder::debugPrintMetadataSummary() {
  if (size_ < sizeof(MetadataHeader)) {
    std::cerr << "ERROR: Buffer too small to contain metadata header\n";
    return;
  }

  std::ostringstream out;

  // --- Read header fields ---
  out << "---- ProductMetadata Debug Summary ----\n";
  out << "Header:\n";
  out << "  Product count:           " << header().productCount << "\n";
  out << "  Serialized buffer size:  " << header().serializedBufferSize << " bytes\n";
  out << "  Flags:";
  if (hasMissing())
    out << " Missing";
  if (hasSerialized())
    out << " Serialized";
  if (hasTrivialCopy())
    out << " TrivialCopy";
  out << "\n\n";

  // store the current offset and rewind the buffer
  size_t offset = readOffset_;
  readOffset_ = sizeof(MetadataHeader);

  // --- Parse product entries ---
  size_t count = 0;
  size_t numMissing = 0;
  size_t numSerialized = 0;
  size_t numTrivial = 0;

  try {
    while (count < static_cast<size_t>(header().productCount)) {
      ProductMetadata meta = getNext();
      ++count;

      out << "Product #" << count << ": ";

      switch (meta.kind) {
        case ProductMetadata::Kind::Missing:
          ++numMissing;
          out << "Missing\n";
          break;

        case ProductMetadata::Kind::Serialized:
          ++numSerialized;
          out << "Serialized, size = " << meta.sizeMeta << "\n";
          break;

        case ProductMetadata::Kind::TrivialCopy:
          ++numTrivial;
          out << "TrivialCopy, size = " << meta.sizeMeta << "\n";
          break;
      }
    }
  } catch (const std::exception& e) {
    out << "\nERROR while parsing metadata: " << e.what() << "\n";
  }

  if (count != static_cast<size_t>(header().productCount)) {
    out << "\nWARNING: Parsed " << count << " entries, but header says " << header().productCount << "\n";
  }

  out << "\n----------------------------------------\n";
  out << "Total entries parsed:   " << count << "\n";
  out << "  Missing:              " << numMissing << "\n";
  out << "  Serialized:           " << numSerialized << "\n";
  out << "  TrivialCopy:          " << numTrivial << "\n";
  out << "Total metadata size:    " << size_ << " bytes\n";

  std::cerr << out.str() << std::flush;

  // restore the current offset
  readOffset_ = offset;
}

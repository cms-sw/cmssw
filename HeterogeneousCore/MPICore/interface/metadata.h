#ifndef HeterogeneousCore_MPICore_interface_metadata_h
#define HeterogeneousCore_MPICore_interface_metadata_h

// C++ standard library headers
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <span>
#include <stdexcept>
#include <variant>
#include <vector>

// MPI headers
#include <mpi.h>

enum ProductFlags : uint8_t {
  HasMissing = 1 << 0,
  HasSerialized = 1 << 1,
  HasTrivialCopy = 1 << 2,
};

struct ProductMetadata {
  enum class Kind : uint8_t { Missing = 0, Serialized = 1, TrivialCopy = 2 };

  Kind kind;
  size_t sizeMeta = 0;
  const uint8_t* trivialCopyOffset = nullptr;  // Only valid if kind == TrivialCopy
};

class ProductMetadataBuilder {
public:
  ProductMetadataBuilder();
  explicit ProductMetadataBuilder(size_t productCount);
  ~ProductMetadataBuilder();

  // No copy
  ProductMetadataBuilder(const ProductMetadataBuilder&) = delete;
  ProductMetadataBuilder& operator=(const ProductMetadataBuilder&) = delete;

  // Move
  ProductMetadataBuilder(ProductMetadataBuilder&& other) noexcept;
  ProductMetadataBuilder& operator=(ProductMetadataBuilder&& other) noexcept;

  // Sender-side: pre-allocate
  void reserve(size_t bytes);

  // set or reset number of products. will fail if not set called before sending
  void setProductCount(size_t prod_num) { productCount_ = prod_num; }
  void setHeader();

  // Sender API
  void addMissing();
  void addSerialized(size_t size);
  void addTrivialCopy(const std::byte* buffer, size_t size);

  const uint8_t* data() const;
  uint8_t* data();
  size_t size() const;
  std::span<const uint8_t> buffer() const;

  // Receiver-side
  void receiveMetadata(int src, int tag, MPI_Comm comm);

  // Not memory safe for trivial copy products.
  // Please make sure that ProductMetadataBuilder lives longer than returned ProductMetadata
  ProductMetadata getNext();

  int64_t productCount() const { return productCount_; }
  int64_t serializedBufferSize() const { return serializedBufferSize_; }
  bool hasMissing() const { return productFlags_ & HasMissing; }
  bool hasSerialized() const { return productFlags_ & HasSerialized; }
  bool hasTrivialCopy() const { return productFlags_ & HasTrivialCopy; }

  void debugPrintMetadataSummary() const;

private:
  uint8_t* buffer_;
  size_t capacity_;
  size_t size_;
  size_t readOffset_;
  const size_t headerSize_ = 13;         // fixed header that stores global info (count, flags, serialized size)
  const size_t maxMetadataSize_ = 1024;  // default size for buffer initialization. Must fit any metadata
  int serializedBufferSize_ = 0;
  uint8_t productFlags_ = 0;
  int64_t productCount_ = 0;

  void resizeBuffer(size_t newCap);
  void ensureCapacity(size_t needed);

  void appendBytes(const std::byte* src, size_t size);

  template <typename T>
  void append(T value) {
    static_assert(std::is_trivially_copyable_v<T>);
    ensureCapacity(sizeof(T));
    std::memcpy(buffer_ + size_, &value, sizeof(T));
    size_ += sizeof(T);
  }

  template <typename T>
  T consume() {
    static_assert(std::is_trivially_copyable_v<T>);
    if (readOffset_ + sizeof(T) > size_)
      throw std::runtime_error("Buffer underflow");
    T val;
    std::memcpy(&val, buffer_ + readOffset_, sizeof(T));
    readOffset_ += sizeof(T);
    return val;
  }
};

#endif  // HeterogeneousCore_MPICore_interface_metadata_h

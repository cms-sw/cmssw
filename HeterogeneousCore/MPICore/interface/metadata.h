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

struct MetadataHeader {
  int16_t productCount = 0;
  uint8_t productFlags = 0;
  int32_t serializedBufferSize = 0;
};

static_assert(sizeof(MetadataHeader) == 8, "Wrong MPI MetadataHeader size - expected to be 8 bytes.");

struct ProductMetadata {
  enum class Kind : uint8_t { Missing = 0, Serialized = 1, TrivialCopy = 2 };

  Kind kind;
  uint64_t sizeMeta = 0;
  const uint8_t* trivialCopyOffset = nullptr;  // Only valid if kind == TrivialCopy
};

class ProductMetadataBuilder {
public:
  ProductMetadataBuilder();
  explicit ProductMetadataBuilder(int16_t productCount);
  ~ProductMetadataBuilder();

  // Not copyable
  ProductMetadataBuilder(const ProductMetadataBuilder&) = delete;
  ProductMetadataBuilder& operator=(const ProductMetadataBuilder&) = delete;

  // Movable
  ProductMetadataBuilder(ProductMetadataBuilder&& other) noexcept;
  ProductMetadataBuilder& operator=(ProductMetadataBuilder&& other) noexcept;

  // Sender-side: pre-allocate
  void reserve(size_t bytes);

  // set or reset number of products. will fail if not set called before sending
  void setProductCount(int16_t prod_num) { header().productCount = prod_num; }

  // Sender API
  void addMissing();
  void addSerialized(size_t size);
  void addTrivialCopy(const std::byte* buffer, size_t size);

  const uint8_t* data() const { return buffer_; }

  uint8_t* data() { return buffer_; }

  size_t size() const { return size_; }

  std::span<const uint8_t> buffer() const { return {buffer_, size_}; }

  // Receiver-side
  void receiveMetadata(int src, int tag, MPI_Comm comm);

  // Not memory safe for trivial copy products.
  // Please make sure that ProductMetadataBuilder lives longer than returned ProductMetadata
  ProductMetadata getNext();

  int16_t productCount() const { return header().productCount; }
  int32_t serializedBufferSize() const { return header().serializedBufferSize; }
  bool hasMissing() const { return header().productFlags & HasMissing; }
  bool hasSerialized() const { return header().productFlags & HasSerialized; }
  bool hasTrivialCopy() const { return header().productFlags & HasTrivialCopy; }

  // non-const because it temporarily modifies the internal state of the object, before restoring it
  void debugPrintMetadataSummary();

private:
  MetadataHeader& header();
  MetadataHeader const& header() const;

  void appendBytes(const std::byte* src, size_t size);
  void consumeBytes(std::byte* dst, size_t size);

  template <typename T>
    requires std::is_trivially_copyable_v<T>
  void append(const T& value) {
    appendBytes(reinterpret_cast<const std::byte*>(&value), sizeof(T));
  }

  template <typename T>
    requires std::is_trivially_copyable_v<T>
  T consume() {
    T value;
    consumeBytes(reinterpret_cast<std::byte*>(&value), sizeof(T));
    return value;
  }

  static constexpr size_t maxMetadataSize_ = 1024;  // default size for buffer initialization. Must fit any metadata

  uint8_t* buffer_;
  size_t capacity_;
  size_t size_;
  size_t readOffset_;
};

#endif  // HeterogeneousCore_MPICore_interface_metadata_h

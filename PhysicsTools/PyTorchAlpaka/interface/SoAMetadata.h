#ifndef PhysicsTools_PyTorchAlpaka_interface_SoAMetadata_h
#define PhysicsTools_PyTorchAlpaka_interface_SoAMetadata_h

#include <cassert>
#include <cmath>
#include <cstddef>
#include <map>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include <any>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <ATen/core/ScalarType.h>

#include "PhysicsTools/PyTorch/interface/TorchCompat.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/ROCmSerialSyncHandle.h"

namespace cms::torch::alpakatools {

  using namespace cms::soa;
  using namespace cms::alpakatools;

  template <typename T, typename... Others>
  concept SameTypes = (std::same_as<T, Others> && ...);

  template <typename T>
  inline static ::torch::ScalarType get_type() {
    return ::torch::CppTypeToScalarType<T>();
  }

  // Wrapper struct to merge info about scalar columns and multidimensional eigen columns
  struct Columns {
    std::vector<int> columns;

    // Empty constructor, to fill iteratively
    Columns() {}

    // Constructor for scalar columns
    Columns(int columns_) { columns.push_back(columns_); }

    // Constructor for multidimensional eigen columns
    Columns(const std::vector<int>& columns_) : columns(columns_) {}
    Columns(std::vector<int>&& columns_) : columns(std::move(columns_)) {}

    size_t size() const { return columns.size(); }
    int operator[](int i) const { return columns[i]; }
    void push(int i) { columns.push_back(i); }
    int n_cols() const { return columns.at(0); }
    int dims() const {
      int dim = 1;
      for (const int d : columns)
        dim *= d;
      return dim / n_cols();
    }
  };

  // Block of SoA Columns with same type and element size.
  // Calculates size and stride and stores torch type.
  struct Block {
    // Constructor for columns and eigen columns
    Block(const int nElements,
          const size_t alignment,
          const void* ptr,
          const Columns& columns,
          const ::torch::ScalarType type,
          const size_t bytes)
        : ptr_(ptr), type_(type), bytes_(bytes), alignment_(alignment) {
      stride_ = create_stride(nElements, alignment, columns, bytes);
      size_ = create_size(nElements, columns);
    };

    // Constructor for scalar columns
    Block(const int nElements,
          const size_t alignment,
          const void* ptr,
          const ::torch::ScalarType type,
          const size_t bytes)
        : ptr_(ptr), type_(type), bytes_(bytes), alignment_(alignment) {
      stride_ = create_stride(nElements, alignment, 1, bytes, true);
      size_ = create_size(nElements, 1);
    };
    virtual ~Block() = default;

    static int get_elems_per_column(const int nElements, const size_t alignment, const size_t bytes) {
      int per_bunch = alignment / bytes;
      int bunches = std::ceil(1.0 * nElements / per_bunch);
      return bunches * per_bunch;
    }

    size_t alignment() const { return alignment_; }
    ::torch::ScalarType type() const { return type_; }
    size_t bytes() const { return bytes_; }
    const std::vector<long int>& size() const { return size_; }
    const std::vector<long int>& stride() const { return stride_; }
    virtual const void* ptr() const { return ptr_; }
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
    virtual void copyToHost(alpaka_rocm_async::Queue& queue) = 0;
    virtual void copyToDevice(alpaka_rocm_async::Queue& queue) = 0;
#endif

  private:
    static std::vector<long int> create_size(const int nElements, const Columns& columns) {
      std::vector<long int> size(columns.size() + 1);
      size[0] = nElements;
      std::copy(columns.columns.begin(), columns.columns.end(), size.begin() + 1);
      if (columns.size() > 1 && columns[0] == 1) {
        size.erase(size.begin() + 1);
      }

      return size;
    }

    static std::vector<long int> create_stride(const int nElements,
                                               const size_t alignment,
                                               const Columns& columns,
                                               const size_t bytes,
                                               const bool is_scalar = false) {
      int N = columns.size() + 1;
      std::vector<long int> stride(N);

      int per_bunch = alignment / bytes;
      int bunches = std::ceil(1.0 * nElements / per_bunch);

      if (!is_scalar)
        stride[0] = 1;
      else {
        // Jump no element per row, to fill with scalar value
        stride[0] = 0;
        bunches = 1;
      }
      stride[std::min(2, N - 1)] = bunches * per_bunch;

      // eigen are stored in column major, but still for every column.
      if (N > 2) {
        for (int i = 3; i < N; i++) {
          stride[i] = stride[i - 1] * columns[i - 2];
        }
        stride[1] = stride[N - 1] * columns[N - 2];
        if (columns[0] == 1) {
          stride.erase(stride.begin() + 1);
        }
      }
      return stride;
    }

    std::vector<long int> stride_;
    std::vector<long int> size_;

    const void* ptr_;
    const ::torch::ScalarType type_;
    const size_t bytes_;
    const size_t alignment_;
  };

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
  template <typename T>
  struct BlockROCm : public Block {
    BlockROCm(const int nElements,
              const size_t alignment,
              const void* ptr,
              const Columns& columns,
              ::torch::ScalarType type,
              const size_t bytes)
        : Block(nElements, alignment, ptr, columns, type, bytes),
          rocm_serial_sync_handle_(
              ptr, columns.n_cols(), get_elems_per_column(nElements, alignment, bytes) * columns.dims()) {}

    // Constructor for scalar columns
    BlockROCm(const int nElements, const size_t alignment, const void* ptr, ::torch::ScalarType type, const size_t bytes)
        : Block(nElements, alignment, ptr, type, bytes), rocm_serial_sync_handle_(ptr, 1, 1) {}

    const void* ptr() const override { return rocm_serial_sync_handle_.ptr(); }
    void copyToHost(alpaka_rocm_async::Queue& queue) override { rocm_serial_sync_handle_.copyToHost(queue); }
    void copyToDevice(alpaka_rocm_async::Queue& queue) override { rocm_serial_sync_handle_.copyToDevice(queue); }

  private:
    alpaka_rocm_async::torch::ROCmSerialSyncHandle<T> rocm_serial_sync_handle_;
  };
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED

  // Metadata for SOA split into multiple blocks.
  // An order for the resulting tensors can be defined.
  struct SoAMetadata {
  private:
    std::map<std::string, std::shared_ptr<Block>> blocks;

    inline static std::vector<int> standard_order(int size) {
      std::vector<int> order(size);
      for (int i = 0; i < size; i++) {
        order[i] = i;
      }
      return order;
    }

    template <typename T, typename... Others>
    bool check_location(int elements, const T* column, const T* other_column, Others... others) {
      return check_location(elements, other_column, others...) && (column + elements) == other_column;
    }

    template <typename T>
    bool check_location(int elements, const T* column, const T* other_column) {
      return (column + elements) == other_column;
    }

    template <typename T>
    bool check_location(int elements, const T* column) {
      return true;
    }

  public:
    // Order of resulting tensor list
    std::vector<std::string> order;
    int nElements;
    int nBlocks;

    SoAMetadata(int nElements_) : nElements(nElements_), nBlocks(0) {}

    // Eigen columns
    template <typename SoALayout, typename T, typename... Others>
      requires(SameTypes<typename T::ValueType, typename Others::ValueType...> && T::columnType == SoAColumnType::eigen)
    void append_block(const std::string& name,
                      int nElements_,
                      std::tuple<T, cms::soa::size_type> column,
                      std::tuple<Others, cms::soa::size_type>... others) {
      using ScalarType = typename T::ScalarType;
      auto [ptr, stride] = std::get<0>(column).tupleOrPointer();
      int elems = Block::get_elems_per_column(nElements_, SoALayout::alignment, sizeof(ScalarType));
      assert(check_location(elems * T::ValueType::RowsAtCompileTime * T::ValueType::ColsAtCompileTime,
                            ptr,
                            std::get<0>(std::get<0>(others).tupleOrPointer())...));

      // initialize Columns with Rows for each column
      Columns col{{1 + sizeof...(Others), T::ValueType::RowsAtCompileTime}};
      // If this is a multi-column Eigen::Matrix, push Cols as well
      if (T::ValueType::ColsAtCompileTime > 1) {
        col.push(T::ValueType::ColsAtCompileTime);
      }

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
      blocks.try_emplace(name,
                         std::make_shared<BlockROCm<ScalarType>>(
                             nElements_, SoALayout::alignment, ptr, col, get_type<ScalarType>(), sizeof(ScalarType)));
#else
      blocks.try_emplace(name,
                         std::make_shared<Block>(
                             nElements_, SoALayout::alignment, ptr, col, get_type<ScalarType>(), sizeof(ScalarType)));
#endif
      order.push_back(name);
      nBlocks += 1;
    }

    // Append a block based on a typed pointer and a column object.
    template <typename SoALayout, typename T, typename... Others>
      requires(SameTypes<typename T::ScalarType, typename Others::ScalarType...> &&
               T::columnType == SoAColumnType::column)
    void append_block(const std::string& name,
                      int nElements_,
                      std::tuple<T, cms::soa::size_type> column,
                      std::tuple<Others, cms::soa::size_type>... others) {
      using ScalarType = typename T::ScalarType;
      int elems = Block::get_elems_per_column(nElements_, SoALayout::alignment, sizeof(ScalarType));
      assert(check_location(elems, std::get<0>(column).tupleOrPointer(), std::get<0>(others).tupleOrPointer()...));

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
      blocks.try_emplace(name,
                         std::make_shared<BlockROCm<ScalarType>>(nElements_,
                                                                 SoALayout::alignment,
                                                                 std::get<0>(column).tupleOrPointer(),
                                                                 1 + sizeof...(Others),
                                                                 get_type<ScalarType>(),
                                                                 sizeof(ScalarType)));
#else
      blocks.try_emplace(name,
                         std::make_shared<Block>(nElements_,
                                                 SoALayout::alignment,
                                                 std::get<0>(column).tupleOrPointer(),
                                                 1 + sizeof...(Others),
                                                 get_type<ScalarType>(),
                                                 sizeof(ScalarType)));
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED
      order.push_back(name);
      nBlocks += 1;
    }

    // Scalar columns are broadcasted
    template <typename SoALayout, SoAColumnType col_type, typename T>
      requires(std::is_arithmetic_v<T> && col_type == SoAColumnType::scalar)
    void append_block(const std::string& name,
                      int nElements_,
                      std::tuple<SoAParametersImpl<col_type, T>, cms::soa::size_type> column) {
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
      blocks.try_emplace(
          name,
          std::make_shared<BlockROCm<T>>(
              nElements_, SoALayout::alignment, std::get<0>(column).tupleOrPointer(), get_type<T>(), sizeof(T)));
#else
      blocks.try_emplace(
          name,
          std::make_shared<Block>(
              nElements_, SoALayout::alignment, std::get<0>(column).tupleOrPointer(), get_type<T>(), sizeof(T)));
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED
      order.push_back(name);
      nBlocks += 1;
    }

    template <typename SoALayout, typename T, typename... Others>
      requires(SameTypes<typename T::ValueType, typename Others::ValueType...> && T::columnType == SoAColumnType::eigen)
    void append_block(const std::string& name,
                      std::tuple<T, cms::soa::size_type> column,
                      std::tuple<Others, cms::soa::size_type>... others) {
      append_block<SoALayout, T, Others...>(name, nElements, column, others...);
    }

    template <typename SoALayout, typename T, typename... Others>
      requires(SameTypes<typename T::ScalarType, typename Others::ScalarType...> &&
               T::columnType == SoAColumnType::column)
    void append_block(const std::string& name,
                      std::tuple<T, cms::soa::size_type> column,
                      std::tuple<Others, cms::soa::size_type>... others) {
      append_block<SoALayout, T, Others...>(name, nElements, column, others...);
    }

    template <typename SoALayout, SoAColumnType col_type, typename T>
      requires(std::is_arithmetic_v<T> && col_type == SoAColumnType::scalar)
    void append_block(const std::string& name, std::tuple<SoAParametersImpl<col_type, T>, cms::soa::size_type> column) {
      append_block<SoALayout, col_type, T>(name, nElements, column);
    }

    // The order is defined by the order append_block is called.
    // It can be changed by passing a vector of the block names afterwards.
    // All blocks have to be mentioned.
    void change_order(const std::vector<std::string>& new_order) { order = new_order; }
    void change_order(std::vector<std::string>&& new_order) { order = std::move(new_order); }

    inline const Block& operator[](const std::string& key) const { return *blocks.at(key); }

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
    void copyToHost(alpaka_rocm_async::Queue& queue) {
      for (const auto& name : order)
        blocks.at(name)->copyToHost(queue);
    }

    void copyToDevice(alpaka_rocm_async::Queue& queue) {
      for (const auto& name : order)
        blocks.at(name)->copyToDevice(queue);
    }
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED
  };

  // Metadata to run model with input SOA and fill output SOA.
  class ModelMetadata {
  public:
    SoAMetadata input;
    SoAMetadata output;

    // Used in model class to correctly choose multi or single output conversion
    bool multi_head;

    ModelMetadata(SoAMetadata& input_, SoAMetadata& output_, bool multi_head_ = false)
        : input(input_), output(output_), multi_head(multi_head_) {}

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
    // For AMD CPU fallback both inputs and outputs are copied to host
    void copyToHost(alpaka_rocm_async::Queue& queue) {
      input.copyToHost(queue);
      output.copyToHost(queue);
      // explicit synchronize to ensure data is in place before inference
      alpaka::wait(queue);
    }

    // For AMD CPU fallback only outputs are copied to device, no need to copy inputs back
    void copyToDevice(alpaka_rocm_async::Queue& queue) {
      output.copyToDevice(queue);
      // no need to explicitly synchronize, rely on implicit synchronization mechanism in framework
      // alpaka::wait(queue);
    }
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED
  };

}  // namespace cms::torch::alpakatools

#endif  // PhysicsTools_PyTorchAlpaka_interface_SoAMetadata_h

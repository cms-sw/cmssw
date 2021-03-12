#ifndef HeterogeneousCore_MPICore_interface_serialization_h
#define HeterogeneousCore_MPICore_interface_serialization_h

#include <cstddef>
#include <iomanip>
#include <memory>
#include <ostream>
#include <utility>

#include "DataFormats/Common/interface/WrapperBase.h"

namespace io {

  // io::unique_buffer manages a memory buffer of run-time fixed size;
  // the underlying memory is release when the io::unique_buffer is destroyed.
  class unique_buffer {
  public:
    // default constructor
    unique_buffer();

    // allocate a new buffer of the specified size
    unique_buffer(size_t size);

    // adopt an existing memory buffer of the specified size
    unique_buffer(std::byte* data, size_t size);
    unique_buffer(std::unique_ptr<std::byte>&& data, size_t size);

    // default copy and move constructors and assignment operators
    unique_buffer(unique_buffer const&) = delete;
    unique_buffer(unique_buffer&&) = default;
    unique_buffer& operator=(unique_buffer const&) = delete;
    unique_buffer& operator=(unique_buffer&&) = default;

    // default destructor, releasing the owned memory
    ~unique_buffer() = default;

    // access to the underlying memory
    std::byte const* data() const;
    std::byte* data();

    // release ownership of the underlying memory
    std::byte* release();

    // return the size of of the buffer
    size_t size() const;

  private:
    std::unique_ptr<std::byte> data_;
    size_t size_;
  };

  // dump the content of a buffer, using the same format as `hd`
  template <typename T>
  T& operator<<(T& out, unique_buffer const& buffer) {
    auto data = reinterpret_cast<const char*>(buffer.data());
    auto size = buffer.size();
    unsigned int l = 0;
    for (; l < size / 16; ++l) {
      out << std::setw(8) << std::setfill('0') << std::hex << l * 16 << std::dec;
      for (unsigned int i = l * 16; i < (l + 1) * 16; ++i) {
        out << ((i % 8 == 0) ? "  " : " ");
        out << std::setw(2) << std::setfill('0') << std::hex << (((int)data[i]) & 0xff) << std::dec;
      }
      out << "  |";
      for (unsigned int i = l * 16; i < (l + 1) * 16; ++i) {
        out << (char)(std::isprint(data[i]) ? data[i] : '.');
      }
      out << "|\n";
    }
    if (size % 16 != 0) {
      out << std::setw(8) << std::setfill('0') << std::hex << l * 16 << std::dec;
      for (unsigned int i = l * 16; i < size; ++i) {
        out << ((i % 8 == 0) ? "  " : " ");
        out << std::setw(2) << std::setfill('0') << std::hex << (((int)data[i]) & 0xff) << std::dec;
      }
      for (unsigned int i = size; i < (l + 1) * 16; ++i) {
        out << ((i % 8 == 0) ? "    " : "   ");
      }
      out << "  |";
      for (unsigned int i = l * 16; i < size; ++i) {
        out << (char)(std::isprint(data[i]) ? data[i] : '.');
      }
      out << "|\n";
    }
    out << std::setw(8) << std::setfill('0') << std::hex << size << std::dec << '\n';
    return out;
  }

  // serialise a Wrapper<T> into an io::unique_buffer
  io::unique_buffer serialize(edm::WrapperBase const& wrapper);

  // deserialise a Wrapper<T> from an io::unique_buffer and store it in a unique_ptr<WrapperBase>
  std::unique_ptr<edm::WrapperBase> deserialize(io::unique_buffer const& buffer);

}  // namespace io

#endif  // HeterogeneousCore_MPICore_interface_serialization_h

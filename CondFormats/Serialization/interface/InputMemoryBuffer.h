#ifndef CondFormats_Serialization_InputMemoryBuffer_h
#define CondFormats_Serialization_InputMemoryBuffer_h

#include <cstddef>
#include <streambuf>

namespace cond {
  namespace serialization {

    // A read-only stream buffer over serialized data in memory, without copying it.
    // std::stringbuf::pubsetbuf cannot be used for this: its effect is
    // implementation-defined, and libc++ ignores it.
    class InputMemoryBuffer : public std::streambuf {
    public:
      InputMemoryBuffer(const void* data, std::size_t size) {
        char* begin = const_cast<char*>(static_cast<const char*>(data));
        setg(begin, begin, begin + size);
      }
    };

  }  // namespace serialization
}  // namespace cond

#endif

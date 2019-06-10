#include <memory>
#include <utility>

#include <TClass.h>
#include <TBufferFile.h>

#include "HeterogeneousCore/MPICore/interface/serialization.h"

namespace io {

  // default constructor
  unique_buffer::unique_buffer() :
    data_(nullptr),
    size_(0)
  {
  }

  // allocate a new buffer of the specified size
  unique_buffer::unique_buffer(size_t size) :
    data_(new std::byte[size]),
    size_(size)
  {
  }

  // adopt an existing memory buffer of the specified size
  unique_buffer::unique_buffer(std::byte* data, size_t size) :
    data_(data),
    size_(size)
  {
  }

  // adopt an existing memory buffer of the specified size
  unique_buffer::unique_buffer(std::unique_ptr<std::byte> && data, size_t size) :
    data_(std::move(data)),
    size_(size)
  {
  }

  // access to the underlying memory
  std::byte const * unique_buffer::data() const
  {
    return data_.get();
  }

  // access to the underlying memory
  std::byte * unique_buffer::data()
  {
    return data_.get();
  }

  // release ownership of the underlying memory
  std::byte * unique_buffer::release()
  {
    return data_.release();
  }

  // return the size of of the buffer
  size_t unique_buffer::size() const
  {
    return size_;
  }


  // serialise a Wrapper<T> into a char[] via a TBufferFile
  unique_buffer serialize(edm::WrapperBase const& wrapper)
  {
    /* TODO
     * construct the buffer with an initial size based on the wrapped class size ?
     * take into account any offset to/from the edm::WrapperBase base class ?
     */ 
    TBufferFile serializer(TBuffer::kWrite);
    serializer.WriteObjectAny(& wrapper, TClass::GetClass(wrapper.wrappedTypeInfo()));

    size_t      size = serializer.Length();
    std::byte * data = reinterpret_cast<std::byte *>(serializer.Buffer());
    serializer.DetachBuffer();

    return unique_buffer(data, size);
  }

  std::unique_ptr<edm::WrapperBase> deserialize(unique_buffer const& buffer)
  {
    TBufferFile deserializer(TBuffer::kRead, buffer.size(), (void *) buffer.data(), false);

    /* TODO try different versions:
     * does not work ?
    return std::unique_ptr<edm::WrapperBase>(reinterpret_cast<edm::WrapperBase *>(deserializer.ReadObjectAny(edmWrapperBaseClass)));
     * works but maybe not always ?
    return std::unique_ptr<edm::WrapperBase>(reinterpret_cast<edm::WrapperBase *>(deserializer.ReadObjectAny(nullptr)));
     * not useful ?
    return std::unique_ptr<edm::WrapperBase>(reinterpret_cast<edm::WrapperBase *>(deserializer.ReadObjectAny(rootType)));
     * not useful ?
    static TClass const* edmWrapperBaseClass = TClass::GetClass(typeid(edm::WrapperBase));
    TClass * rootType = wrappedType.getClass();
    int offset = rootType->GetBaseClassOffset(edmWrapperBaseClass);
    return std::unique_ptr<edm::WrapperBase>(reinterpret_cast<edm::WrapperBase *>(reinterpret_cast<char *>(deserializer.ReadObjectAny(rootType)) + offset));
     */
    return std::unique_ptr<edm::WrapperBase>(reinterpret_cast<edm::WrapperBase *>(deserializer.ReadObjectAny(nullptr)));
  }

}

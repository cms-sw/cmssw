#ifndef CondCore_DBCommon_TBufferBlobStreamer_h
#define CondCore_DBCommon_TBufferBlobStreamer_h

#include <cstddef>
#include "POOLCore/IBlobStreamingService.h"
#include "CoralBase/Blob.h"
#include "Reflex/Type.h"
#include "TClass.h"

namespace cond {
  class TBufferBlobTypeInfo
  {
  public:
    TBufferBlobTypeInfo( const ROOT::Reflex::Type& type );

    /// length of the plain C array (zero otherwise)
    std::size_t m_arraySize;

    /// The class as seen by the ROOT streaming services
    TClass *m_class;

    /// the primitive C++ type if m_class is unset
    unsigned int m_primitive;
  };

  class TBufferBlobWriter : virtual public pool::IBlobWriter
  {
  public:
    /// Constructor
    explicit TBufferBlobWriter( const ROOT::Reflex::Type& type );

    /// Empty destructor
    virtual ~TBufferBlobWriter();

    /// Streams an object an returns by constant reference the underlying blob
    const coral::Blob& write( const void* addressOfInputData );

  private:
    /// The type information as seen by the blob streamer
    TBufferBlobTypeInfo m_type;

    /// The blob data
    coral::Blob m_blob;
  };


  class TBufferBlobReader : virtual public pool::IBlobReader
  {
  public:
    /// Constructor
    TBufferBlobReader( const ROOT::Reflex::Type& type );

    /// Empty destructor
    virtual ~TBufferBlobReader();

    /// Reads an object from a Blob and returns the starting address of the object
    void read( const coral::Blob& blobData,
               void* containerAddress ) const;

  private:
    /// The type information as seen by the blob streamer
    TBufferBlobTypeInfo m_type;
  };

}

#endif // COND_TBUFFERBLOBSTREAMER_H

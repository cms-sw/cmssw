#ifndef CondCore_DBCommon_TBufferBlobStreamer_h
#define CondCore_DBCommon_TBufferBlobStreamer_h

#include "CondCore/ORA/interface/IBlobStreamingService.h"
//
#include <cstddef>
//
#include "CoralBase/Blob.h"

#include "FWCore/Utilities/interface/TypeWithDict.h"

#include "TClass.h"

namespace cond {
  class TBufferBlobTypeInfo {
  public:
    TBufferBlobTypeInfo( const edm::TypeWithDict& type );

    /// length of the plain C array (zero otherwise)
    std::size_t m_arraySize;

    /// The class as seen by the ROOT streaming services
    TClass *m_class;

    /// the primitive C++ type if m_class is unset
    unsigned int m_primitive;
  };

  class TBufferBlobStreamingService : virtual public ora::IBlobStreamingService {
    public:
    TBufferBlobStreamingService();
    
    virtual ~TBufferBlobStreamingService();

    boost::shared_ptr<coral::Blob> write( const void* addressOfInputData,  edm::TypeWithDict const & classDictionary, bool useCompression=false );

    void read( const coral::Blob& blobData, void* addressOfContainer,  edm::TypeWithDict const & classDictionary );

  };
  
}

#endif // COND_TBUFFERBLOBSTREAMER_H

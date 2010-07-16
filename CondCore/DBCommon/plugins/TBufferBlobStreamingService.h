#ifndef CondCore_DBCommon_TBufferBlobStreamer_h
#define CondCore_DBCommon_TBufferBlobStreamer_h

#include "CondCore/ORA/interface/IBlobStreamingService.h"
//
#include <cstddef>
//
#include "CoralBase/Blob.h"
#include "Reflex/Type.h"
#include "TClass.h"
#include "RVersion.h"

#if ROOT_VERSION_CODE < ROOT_VERSION(5,19,0)
 typedef ROOT::Reflex::Type TypeH;
#else
 typedef Reflex::Type TypeH;
#endif

namespace cond {
  class TBufferBlobTypeInfo {
  public:
    TBufferBlobTypeInfo( const TypeH& type );

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

    boost::shared_ptr<coral::Blob> write( const void* addressOfInputData, const TypeH& classDictionary );

    void read( const coral::Blob& blobData, void* addressOfContainer, const TypeH& classDictionary );

  };
  
}

#endif // COND_TBUFFERBLOBSTREAMER_H

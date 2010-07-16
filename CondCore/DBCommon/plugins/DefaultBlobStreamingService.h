#ifndef COND_DEFAULTBLOBSTREAMINGSERVICE_H
#define COND_DEFAULTBLOBSTREAMINGSERVICE_H

#include "Reflex/Type.h"
#include "CoralBase/Blob.h"
#include "CondCore/ORA/interface/IBlobStreamingService.h"

#if ROOT_VERSION_CODE < ROOT_VERSION(5,19,0)
typedef ROOT::Reflex::Type TypeH;
#else
typedef Reflex::Type TypeH;
#endif

namespace cond {
  // prerequisite check for supported dictionary type
  
  struct PrimitiveContainerDictPrereq {
      bool operator () ( const TypeH& classDictionary );
  };

  class DefaultBlobStreamingService: virtual public ora::IBlobStreamingService {

    public:
    DefaultBlobStreamingService();
    
    virtual ~DefaultBlobStreamingService();

    boost::shared_ptr<coral::Blob> write( const void* addressOfInputData, const TypeH& classDictionary );

    void read( const coral::Blob& blobData, void* addressOfContainer, const TypeH& classDictionary );

  };

}

#endif // COND_DEFAULTBLOBSTREAMINGSERVICE_H

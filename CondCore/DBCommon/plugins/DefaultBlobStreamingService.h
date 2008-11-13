#ifndef  CondCore_DBCommon_DefaultBlobStreamingService_h
#define  CondCore_DBCommon_DefaultBlobStreamingService_h
#include "BlobStreamingService.h"
#include "PrimitivesContainerStreamer.h"
namespace cond{ 
  class DefaultBlobStreamingService :
    public BlobStreamingService<BlobWriter,BlobReader,PrimitiveContainerDictPrereq > {
    public:
    /// Standard Constructor with a component key
    explicit DefaultBlobStreamingService( const std::string& key ):parent(key){}
  };
}
#endif

#ifndef  CondCore_DBCommon_TBufferBlobStreamingService_h
#define  CondCore_DBCommon_TBufferBlobStreamingService_h
#include "BlobStreamingService.h"
#include "TBufferBlobStreamer.h"

namespace cond {
  class TBufferBlobStreamingService:public BlobStreamingService<TBufferBlobWriter,TBufferBlobReader> {
  public:
    /// Standard Constructor with a component key
    explicit TBufferBlobStreamingService( const std::string& key ):parent(key){}
  };
}
#endif

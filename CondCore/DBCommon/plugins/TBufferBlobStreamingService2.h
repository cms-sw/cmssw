#ifndef  CondCore_DBCommon_TBufferBlobStreamingService2_h
#define  CondCore_DBCommon_TBufferBlobStreamingService2_h
#include "BlobStreamingService.h"
#include "TBufferBlobStreamer.h"

namespace cond {
  class TBufferBlobStreamingService2:public BlobStreamingService<TBufferBlobWriter,TBufferBlobReader> {
  public:
    /// Standard Constructor with a component key
    explicit TBufferBlobStreamingService2( const std::string& key ):parent(key){}
  };
}
#endif

#ifndef DataFormats_Common_CacheStreamers_h
#define DataFormats_Common_CacheStreamers_h

#include "TClassStreamer.h"
#include "TClassRef.h"
class TBuffer;

namespace edm {
  class ConstPtrCacheStreamer : public TClassStreamer {
  public:
    explicit ConstPtrCacheStreamer() : cl_("edm::ConstPtrCache"){}

    void operator() (TBuffer &R__b, void *objp);

  private:
    TClassRef cl_;
  };

  class BoolCacheStreamer : public TClassStreamer {
public:
    explicit BoolCacheStreamer() : cl_("edm::BoolCache"){}
    
    void operator() (TBuffer &R__b, void *objp);
    
private:
    TClassRef cl_;
  };
  
  
  void setCacheStreamers();
}

#endif

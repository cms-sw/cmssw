#include "DataFormats/Common/interface/CacheStreamers.h"
#include "DataFormats/Common/interface/ConstPtrCache.h"
#include "DataFormats/Common/interface/BoolCache.h"
#include "TROOT.h"
class TBuffer;

namespace edm {
  void 
  BoolCacheStreamer::operator()(TBuffer &R__b, void *objp) {
    if (R__b.IsReading()) {
      cl_->ReadBuffer(R__b, objp);
      BoolCache* obj = static_cast<BoolCache *>(objp);
      *obj = false;
    } else {
      cl_->WriteBuffer(R__b, objp);
    }
  }

  void 
  ConstPtrCacheStreamer::operator()(TBuffer &R__b, void *objp) {
    if (R__b.IsReading()) {
      cl_->ReadBuffer(R__b, objp);
      ConstPtrCache* obj = static_cast<ConstPtrCache *>(objp);
      obj->ptr_=0;
    } else {
      cl_->WriteBuffer(R__b, objp);
    }
  }
  
  void setCacheStreamers() {
#if 0    
    TClass *cl = gROOT->GetClass("edm::BoolCache");
    if (cl->GetStreamer() == 0) {
      cl->AdoptStreamer(new BoolCacheStreamer());
    /*} else {
      std::cout <<"ERROR: no edm::BoolCache found"<<std::endl;*/
    }

    cl = gROOT->GetClass("edm::ConstPtrCache");
    if (cl->GetStreamer() == 0) {
      cl->AdoptStreamer(new ConstPtrCacheStreamer());
    /*} else {
      std::cout <<"ERROR: no edm::ConstPtrCache found"<<std::endl;*/
    }
#endif    
  }
}

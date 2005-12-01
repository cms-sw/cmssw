#include "IOPool/Common/interface/RefStreamer.h"
#include "FWCore/EDProduct/interface/RefCore.h"

namespace edm {
  void 
  RefStreamer::operator()(TBuffer &R__b, void *objp) {
    if (R__b.IsReading()) {
      cl_->ReadBuffer(R__b, objp);
      RefCore* obj = static_cast<RefCore *>(objp);
      obj->setProductGetter(prodGetter_);
    } else {
      cl_->WriteBuffer(R__b, objp);
    }
  }

  void SetRefStreamer() {
    TClass *cl = gROOT->GetClass("edm::RefCore");
    if (cl->GetStreamer() == 0) {
      cl->AdoptStreamer(new RefStreamer(0));
    }
  }

  void SetRefStreamer(EDProductGetter const* ep) {
    TClass *cl = gROOT->GetClass("edm::RefCore");
    RefStreamer *st = static_cast<RefStreamer *>(cl->GetStreamer());
    if (st == 0) {
      cl->AdoptStreamer(new RefStreamer(ep));
    } else {
      st->setProductGetter(ep);
    }
  }
}

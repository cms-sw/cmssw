#include "IOPool/Common/interface/RefStreamer.h"
#include "DataFormats/Common/interface/RefCore.h"
#include "FWCore/Utilities/interface/EDMException.h"
class TBuffer;
#include "TROOT.h"

namespace edm {
  void 
  RefStreamer::operator()(TBuffer &R__b, void *objp) {
    if (R__b.IsReading()) {
      cl_->ReadBuffer(R__b, objp);
      RefCore* obj = static_cast<RefCore *>(objp);
      obj->setProductGetter(prodGetter_);
      obj->setProductPtr(0);
    } else {
      RefCore* obj = static_cast<RefCore *>(objp);
      if (obj->isTransient()) {
        throw Exception(errors::InvalidReference,"Inconsistency")
          << "RefStreamer: transient Ref or Ptr cannot be made persistent.";
      }
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

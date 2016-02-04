#include "RefStreamer.h"
#include "DataFormats/Common/interface/RefCore.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "TROOT.h"
#include <assert.h>
#include <ostream>

class TBuffer;

namespace fwlite {
  void 
  RefStreamer::operator()(TBuffer &R__b, void *objp) {
    using edm::RefCore;
    using edm::Exception;
    using edm::errors::InvalidReference;
    typedef RefCore::RefCoreTransients RefCoreTransients;
    if (R__b.IsReading()) {
      cl_->ReadBuffer(R__b, objp);
      RefCoreTransients* obj = static_cast<RefCoreTransients *>(objp);
      obj->setProductGetter(prodGetter_);
      obj->setProductPtr(0);
    } else {
      RefCoreTransients* obj = static_cast<RefCoreTransients *>(objp);
      if (obj->isTransient()) {
        throw Exception(InvalidReference,"Inconsistency")
          << "RefStreamer: transient Ref or Ptr cannot be made persistent.";
      }
      cl_->WriteBuffer(R__b, objp);
    }
  }

  edm::EDProductGetter const* setRefStreamer(edm::EDProductGetter const* ep) {
    using namespace edm;
    static TClass *cl = gROOT->GetClass("edm::RefCore::RefCoreTransients");
    assert(cl);
    RefStreamer *st = static_cast<RefStreamer *>(cl->GetStreamer());
    edm::EDProductGetter const* pOld = 0;
    if (st == 0) {
      cl->AdoptStreamer(new RefStreamer(ep));
    } else {
      pOld = st->setProductGetter(ep);
    }
    return pOld;
  }
}

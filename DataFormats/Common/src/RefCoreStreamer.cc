#include "DataFormats/Common/interface/RefCoreStreamer.h"
#include "DataFormats/Common/interface/RefCore.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "TROOT.h"
#include <ostream>
#include <cassert>
#include <iostream>

namespace edm {

  void
  RefCoreStreamer::operator()(TBuffer &R__b, void *objp) {
    if (R__b.IsReading()) {
      cl_->ReadBuffer(R__b, objp);
      RefCore* obj = static_cast<RefCore *>(objp);
      obj->setProductGetter(prodGetter_);
      //Now ProductGetter and ProductPtr share the same internal pointer
      // so only need to set one
      //obj->setProductPtr(0);
    } else {
      assert("RefCore streamer is obsolete" == 0);
    }
  }


  /*NOTE: This design came from Philippe Canal as the minimum storage (2bytes) we can do but still
   have ROOT call our custom streamer. The trick is to only store the version # and not the class ID.
   The '#if #else #endif' are there because the default choice is known to work for root 5.27-5.28 and
   Philippe believes is unlikely to ever change but the alternate choice is slightly slower but more
   guaranteed to be forwards compatible.
   */
  void
  RefCoreCheckTransientOnWriteStreamer::operator()(TBuffer &R__b, void *objp) {
    typedef RefCore::CheckTransientOnWrite CheckTransientOnWrite;
    if (R__b.IsReading()) {
      //std::cout <<"reading CheckTransientOnWrite"<<std::endl;
#if 1
      Version_t version;
      R__b >> version;
#else
      R__b.ReadVersion();
#endif
      CheckTransientOnWrite* obj = static_cast<CheckTransientOnWrite *>(objp);
      obj->transient_ = false;
      obj->cacheIsProductPtr_ = false;
    } else {
      //std::cout <<"writing CheckTransientOnWrite"<<std::endl;
      TVirtualStreamerInfo* sinfo = cl_->GetStreamerInfo();
      CheckTransientOnWrite* obj = static_cast<CheckTransientOnWrite *>(objp);
      if (obj->transient_) {
        throw Exception(errors::InvalidReference,"Inconsistency")
          << "RefCoreStreamer: transient Ref or Ptr cannot be made persistent.";
      }
#if 1
      R__b << cl_->GetClassVersion();
#else
      R__b.WriteVersion(cl_, kFALSE);
#endif
      R__b.TagStreamerInfo(sinfo);
    }
  }

  void setRefCoreStreamer(bool resetAll) {
    {
      TClass *cl = gROOT->GetClass("edm::RefCore::CheckTransientOnWrite");
      TClassStreamer *st = cl->GetStreamer();
      if (st == 0) {
        cl->AdoptStreamer(new RefCoreCheckTransientOnWriteStreamer());
      }
    }
    EDProductGetter::switchProductGetter(0);
    if (resetAll) {
      TClass *cl = gROOT->GetClass("edm::RefCore");
      if (cl->GetStreamer() != 0) {
        cl->AdoptStreamer(0);
      }
    }
  }

  EDProductGetter const* setRefCoreStreamer(EDProductGetter const* ep) {
    EDProductGetter const* returnValue=0;
    if (ep != 0) {
      TClass *cl = gROOT->GetClass("edm::RefCore::CheckTransientOnWrite");
      TClassStreamer *st = cl->GetStreamer();
      if (st == 0) {
        cl->AdoptStreamer(new RefCoreCheckTransientOnWriteStreamer());
      }
      returnValue = edm::EDProductGetter::switchProductGetter(ep);
    }
    return returnValue;
  }
}

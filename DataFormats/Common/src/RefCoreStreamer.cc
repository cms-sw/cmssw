#include "DataFormats/Common/interface/RefCoreStreamer.h"
#include "DataFormats/Common/interface/RefCore.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "TROOT.h"
#include <ostream>
#include <cassert>

namespace edm {
  void 
  ProductIDStreamer::operator()(TBuffer &R__b, void *objp) {
    if (R__b.IsReading()) {
      UInt_t i0, i1;
      R__b.ReadVersion(&i0, &i1, cl_);
      unsigned int id;
      R__b >> id;
      ProductID pid;
      pid.oldID() = id;
      ProductID* obj = static_cast<ProductID *>(objp);
      *obj = (prodGetter_ ? prodGetter_->oldToNewProductID(pid) : pid);
    } else {
      assert("ProductID streamer is obsolete" == 0);
    }
  }

  void 
  RefCoreStreamer::operator()(TBuffer &R__b, void *objp) {
    if (R__b.IsReading()) {
      cl_->ReadBuffer(R__b, objp);
      RefCore* obj = static_cast<RefCore *>(objp);
      obj->setProductGetter(prodGetter_);
      obj->setProductPtr(0);
    } else {
      assert("RefCore streamer is obsolete" == 0);
    }
  }

  void 
  RefCoreTransientStreamer::operator()(TBuffer &R__b, void *objp) {
    typedef RefCore::RefCoreTransients RefCoreTransients;
    if (R__b.IsReading()) {
      cl_->ReadBuffer(R__b, objp);
      RefCoreTransients* obj = static_cast<RefCoreTransients *>(objp);
      obj->setProductGetter(prodGetter_);
      obj->setProductPtr(0);
    } else {
      RefCoreTransients* obj = static_cast<RefCoreTransients *>(objp);
      if (obj->isTransient()) {
        throw Exception(errors::InvalidReference,"Inconsistency")
          << "RefCoreStreamer: transient Ref or Ptr cannot be made persistent.";
      }
      cl_->WriteBuffer(R__b, objp);
    }
  }

  void setRefCoreStreamer(bool oldFormat) {
    {
      TClass *cl = gROOT->GetClass("edm::RefCore::RefCoreTransients");
      RefCoreTransientStreamer *st = static_cast<RefCoreTransientStreamer *>(cl->GetStreamer());
      if (st == 0) {
        cl->AdoptStreamer(new RefCoreTransientStreamer(0));
      } else {
        st->setProductGetter(0);
      }
    }
    if (oldFormat) {
      TClass *cl = gROOT->GetClass("edm::RefCore");
      if (cl->GetStreamer() != 0) {
        cl->AdoptStreamer(0);
      }
    }
    if (oldFormat) {
      TClass *cl = gROOT->GetClass("edm::ProductID");
      if (cl->GetStreamer() != 0) {
        cl->AdoptStreamer(0);
      }
    }
  }

  void setRefCoreStreamer(EDProductGetter const* ep, bool oldFormat) {
    if (ep != 0) {
      if (oldFormat) {
        TClass *cl = gROOT->GetClass("edm::RefCore");
        RefCoreStreamer *st = static_cast<RefCoreStreamer *>(cl->GetStreamer());
        if (st == 0) {
          cl->AdoptStreamer(new RefCoreStreamer(ep));
        } else {
          st->setProductGetter(ep);
        }
      } else {
        TClass *cl = gROOT->GetClass("edm::RefCore::RefCoreTransients");
        RefCoreTransientStreamer *st = static_cast<RefCoreTransientStreamer *>(cl->GetStreamer());
        if (st == 0) {
          cl->AdoptStreamer(new RefCoreTransientStreamer(ep));
        } else {
          st->setProductGetter(ep);
        }
      }
    }
    if (oldFormat) {
      TClass *cl = gROOT->GetClass("edm::ProductID");
      ProductIDStreamer *st = static_cast<ProductIDStreamer *>(cl->GetStreamer());
      if (st == 0) {
        cl->AdoptStreamer(new ProductIDStreamer(ep));
      } else {
        st->setProductGetter(ep);
      }
    }
  }
}

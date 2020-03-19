#include "DataFormats/Common/interface/RefCoreStreamer.h"
#include "DataFormats/Common/interface/RefCore.h"
#include "DataFormats/Common/interface/RefCoreWithIndex.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/Common/interface/EDProductGetter.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "TBuffer.h"
#include "TClass.h"

namespace edm {

  /*NOTE: This design came from Philippe Canal as the minimum storage (2bytes) we can do but still
   have ROOT call our custom streamer. The trick is to only store the version # and not the class ID.
   The '#if #else #endif' are there because the default choice is known to work for root 5.27-5.28 and
   Philippe believes is unlikely to ever change but the alternate choice is slightly slower but more
   guaranteed to be forwards compatible.
   */

  void RefCoreStreamer::operator()(TBuffer& R__b, void* objp) {
    if (R__b.IsReading()) {
      cl_->ReadBuffer(R__b, objp);
    } else {
      //If transient, throw
      RefCore* obj = static_cast<RefCore*>(objp);
      if (obj->isTransient()) {
        throw Exception(errors::InvalidReference, "Inconsistency")
            << "RefCoreStreamer: transient Ref or Ptr cannot be made persistent.";
      }
#if 1
      R__b << cl_->GetClassVersion();
#else
      R__b.WriteVersion(cl_, kFALSE);
#endif
      //Must match the order the data appears in the class declaration
      const ProductID& id = obj->id();
      R__b << id.processIndex();
      R__b << id.productIndex();
    }
  }

  void RefCoreWithIndexStreamer::operator()(TBuffer& R__b, void* objp) {
    if (R__b.IsReading()) {
      cl_->ReadBuffer(R__b, objp);
    } else {
      //If transient, throw
      RefCoreWithIndex* obj = static_cast<RefCoreWithIndex*>(objp);
      if (obj->isTransient()) {
        throw Exception(errors::InvalidReference, "Inconsistency")
            << "RefCoreStreamer: transient Ref or Ptr cannot be made persistent.";
      }
#if 1
      R__b << cl_->GetClassVersion();
#else
      R__b.WriteVersion(cl_, kFALSE);
#endif
      //Must match the order the data appears in the class declaration
      const ProductID& id = obj->id();
      R__b << id.processIndex();
      R__b << id.productIndex();
      R__b << obj->index();
    }
  }

  TClassStreamer* RefCoreStreamer::Generate() const { return new RefCoreStreamer(*this); }

  TClassStreamer* RefCoreWithIndexStreamer::Generate() const { return new RefCoreWithIndexStreamer(*this); }

  void setRefCoreStreamerInTClass() {
    {
      TClass* tClass = TClass::GetClass("edm::RefCore");
      if (tClass->GetStreamer() == nullptr) {
        tClass->AdoptStreamer(new RefCoreStreamer());
      }
    }
    {
      TClass* tClass = TClass::GetClass("edm::RefCoreWithIndex");
      if (tClass->GetStreamer() == nullptr) {
        tClass->AdoptStreamer(new RefCoreWithIndexStreamer());
      }
    }
  }

  void setRefCoreStreamer(bool) { EDProductGetter::switchProductGetter(nullptr); }

  EDProductGetter const* setRefCoreStreamer(EDProductGetter const* ep) {
    EDProductGetter const* returnValue = nullptr;
    if (ep != nullptr) {
      returnValue = edm::EDProductGetter::switchProductGetter(ep);
    }
    return returnValue;
  }
}  // namespace edm

#ifndef IOPool_Common_TransientStreamer_h
#define IOPool_Common_TransientStreamer_h

#include <string>
#include "TROOT.h"
#include "TClassStreamer.h"
#include "TClassRef.h"
#include "FWCore/Utilities/interface/TypeID.h"
class TBuffer;

namespace edm {
  template <typename T>
  class TransientStreamer : public TClassStreamer {
  public:
    typedef T element_type;
    TransientStreamer();
    void operator() (TBuffer &R__b, void *objp);
  private:
    std::string className_;
    TClassRef cl_;
  };

  template <typename T>
  TransientStreamer<T>::TransientStreamer() :
    className_(TypeID(typeid(T)).className()),
    cl_(className_.c_str())
  {}

  template <typename T>
  void
  TransientStreamer<T>::operator()(TBuffer &R__b, void *objp) {
    if (R__b.IsReading()) {
      cl_->ReadBuffer(R__b, objp);
      // Fill with default constructed object;
      T* obj = static_cast<T *>(objp);
      *obj = T();
    } else {
      cl_->WriteBuffer(R__b, objp);
    }
  }

  template <typename T>
  void
  SetTransientStreamer() {
    TClass *cl = gROOT->GetClass(TypeID(typeid(T)).className().c_str());
    if (cl->GetStreamer() == 0) {
      cl->AdoptStreamer(new TransientStreamer<T>());
    }
  }

  template <typename T>
  void
  SetTransientStreamer(T const&) {
    TClass *cl = gROOT->GetClass(TypeID(typeid(T)).className().c_str());
    if (cl->GetStreamer() == 0) {
      cl->AdoptStreamer(new TransientStreamer<T>());
    }
  }
}

#endif

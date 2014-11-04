#ifndef IOPool_Common_CustomStreamer_h
#define IOPool_Common_CustomStreamer_h

#include <string>
#include "TClass.h"
#include "TClassStreamer.h"
#include "TClassRef.h"
#include "FWCore/Utilities/interface/TypeID.h"
class TBuffer;

namespace edm {
  template <typename T>
  class CustomStreamer : public TClassStreamer {
  public:
    typedef T element_type;
    CustomStreamer();
    void operator() (TBuffer &R__b, void *objp);
  private:
    std::string className_;
    TClassRef cl_;
  };

  template <typename T>
  CustomStreamer<T>::CustomStreamer() :
    className_(TypeID(typeid(T)).className()),
    cl_(className_.c_str())
  {}

  template <typename T>
  void
  CustomStreamer<T>::operator()(TBuffer &R__b, void *objp) {
    if (R__b.IsReading()) {
      cl_->ReadBuffer(R__b, objp);
    } else {
      cl_->WriteBuffer(R__b, objp);
    }
  }

  template <typename T>
  void
  SetCustomStreamer() {
    TClass *cl = TClass::GetClass(TypeID(typeid(T)).className().c_str());
    if (cl->GetStreamer() == 0) {
      cl->AdoptStreamer(new CustomStreamer<T>());
    }
  }

  template <typename T>
  void
  SetCustomStreamer(T const&) {
    TClass *cl = TClass::GetClass(TypeID(typeid(T)).className().c_str());
    if (cl->GetStreamer() == 0) {
      cl->AdoptStreamer(new CustomStreamer<T>());
    }
  }
}

#endif

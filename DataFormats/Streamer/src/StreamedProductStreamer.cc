#include "DataFormats/Streamer/interface/StreamedProductStreamer.h"
#include "DataFormats/Streamer/interface/StreamedProducts.h"

#include "TROOT.h"

namespace edm {
  void 
  StreamedProductStreamer::operator()(TBuffer& R__b, void *objp) {
    if (R__b.IsReading()) {
      cl_->ReadBuffer(R__b, objp);
      StreamedProduct* obj = static_cast<StreamedProduct*>(objp);
      obj->clearClassType();
      obj->setNewClassType();
      if(productstatus::present(obj->status())) {
        obj->allocateForReading();
        obj->classRef()->Streamer(obj->prod(), R__b);
      }
    } else {
      cl_->WriteBuffer(R__b, objp);
      StreamedProduct* obj = static_cast<StreamedProduct*>(objp);
      obj->setNewClassType();
      if(productstatus::present(obj->status())) {
        assert(obj->prod() != 0);
        obj->classRef()->Streamer(obj->prod(), R__b);
      }
    }
  }

  void
  setStreamedProductStreamer() {
    TClass *cl = gROOT->GetClass("edm::StreamedProduct");
    cl->AdoptStreamer(new StreamedProductStreamer());
  }
}

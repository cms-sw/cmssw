#ifndef DataFormats_Streamer_StreamedProductStreamer_h
#define DataFormats_Streamer_StreamedProductStreamer_h

#include "TClassStreamer.h"

namespace edm {
  class StreamedProductStreamer : public TClassStreamer {
  public:
    StreamedProductStreamer() : cl_("edm::StreamedProduct") {}

    void operator() (TBuffer &R__b, void *objp);

  private:
    TClassRef cl_;
  };

  void setStreamedProductStreamer();
}

#endif

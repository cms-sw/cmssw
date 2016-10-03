#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilter.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace RecoPixelVertexing_PixelTrackFitting {
  struct dictionary {
    PixelTrackFilter ptf;
    edm::Wrapper<PixelTrackFilter> wptf;
  };
}

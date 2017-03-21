#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitter.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilter.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace RecoPixelVertexing_PixelTrackFitting {
  struct dictionary {
    PixelFitter pf;
    edm::Wrapper<PixelFitter> wpf;

    PixelTrackFilter ptf;
    edm::Wrapper<PixelTrackFilter> wptf;
  };
}

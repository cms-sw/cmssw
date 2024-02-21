#include "DataFormats/VertexReco/interface/TrackTimeLifeInfo.h"

TrackTimeLifeInfo::TrackTimeLifeInfo()
    : hasSV_(false),
      hasTrack_(false),
      sv_(reco::Vertex()),
      flight_vec_(GlobalVector()),
      ip_vec_(GlobalVector()),
      pca_(GlobalPoint()),
      flight_cov_(GlobalError()),
      pca_cov_(GlobalError()),
      ip_cov_(GlobalError()),
      flightLength_(Measurement1D()),
      ipLength_(Measurement1D()),
      track_(reco::Track()),
      bField_z_(0.){};

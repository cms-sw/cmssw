#ifndef SiPixelPhase1TrackResiduals_h 
#define SiPixelPhase1TrackResiduals_h 
// -*- C++ -*-
// 
// Package:     SiPixelPhase1TrackResiduals
// Class  :     SiPixelPhase1TrackResiduals
//

// Original Author: Marcel Schneider

#include "DQM/SiPixelPhase1Common/interface/SiPixelPhase1Base.h"
#include "Alignment/OfflineValidation/interface/TrackerValidationVariables.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

class SiPixelPhase1TrackResiduals : public SiPixelPhase1Base {
  enum {
    RESIDUAL_X,
    RESIDUAL_Y
  };

  public:
  explicit SiPixelPhase1TrackResiduals(const edm::ParameterSet& conf);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  private:
  TrackerValidationVariables validator;
  edm::EDGetTokenT<reco::VertexCollection> offlinePrimaryVerticesToken_;

  bool applyVertexCut_;
};

#endif

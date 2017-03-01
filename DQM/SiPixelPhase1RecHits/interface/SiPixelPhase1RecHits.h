#ifndef SiPixelPhase1RecHits_h 
#define SiPixelPhase1RecHits_h 
// -*- C++ -*-
// 
// Package:     SiPixelPhase1RecHits
// Class  :     SiPixelPhase1RecHits
//

// Original Author: Marcel Schneider

#include "DQM/SiPixelPhase1Common/interface/SiPixelPhase1Base.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"

class SiPixelPhase1RecHits : public SiPixelPhase1Base {
  enum {
    NRECHITS,
    CLUST_X,
    CLUST_Y,
    ERROR_X,
    ERROR_Y,
    POS,
    CLUSTER_PROB
  };

  public:
  explicit SiPixelPhase1RecHits(const edm::ParameterSet& conf);
  void analyze(const edm::Event&, const edm::EventSetup&);

  private:
  edm::EDGetTokenT<SiPixelRecHitCollection> srcToken_;
};

#endif

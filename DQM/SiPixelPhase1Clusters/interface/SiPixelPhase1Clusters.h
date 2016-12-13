#ifndef SiPixelPhase1Clusters_h 
#define SiPixelPhase1Clusters_h 
// -*- C++ -*-
// 
// Package:     SiPixelPhase1Clusters
// Class  :     SiPixelPhase1Clusters
//

// Original Author: Marcel Schneider

#include "DQM/SiPixelPhase1Common/interface/SiPixelPhase1Base.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

class SiPixelPhase1Clusters : public SiPixelPhase1Base {
  enum {
    CHARGE,
    SIZE,
    NCLUSTERS,
    NCLUSTERSINCLUSIVE,
    EVENTRATE,
    POSITION_B,
    POSITION_F,
    POSITION_XZ,
    POSITION_YZ,
    SIZE_VS_ETA,
    READOUT_CHARGE,
    READOUT_NCLUSTERS
  };

  public:
  explicit SiPixelPhase1Clusters(const edm::ParameterSet& conf);
  void analyze(const edm::Event&, const edm::EventSetup&);

  private:
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > srcToken_;
};

#endif

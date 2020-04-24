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
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

class SiPixelPhase1Clusters : public SiPixelPhase1Base {
  enum {
    CHARGE,
    SIZE,
    SIZEX,
    SIZEY,
    NCLUSTERS,
    NCLUSTERSINCLUSIVE,
    EVENTRATE,
    POSITION_B,
    POSITION_F,
    POSITION_XZ,
    POSITION_YZ,
    SIZE_VS_ETA,
    READOUT_CHARGE,
    READOUT_NCLUSTERS,
    PIXEL_TO_STRIP_RATIO
  };
  // Uncomment to add trigger event flag enumerators
  // Make sure enum corresponds correctly with flags defined in _cfi.py file
  // enum {
  //   FLAG_HLT,
  //   FLAG_L1,
  // }

  public:
  explicit SiPixelPhase1Clusters(const edm::ParameterSet& conf);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  private:
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > pixelSrcToken_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> > stripSrcToken_;
};

#endif

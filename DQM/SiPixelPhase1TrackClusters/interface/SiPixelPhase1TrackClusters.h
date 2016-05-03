#ifndef SiPixelPhase1TrackClusters_h 
#define SiPixelPhase1TrackClusters_h 
// -*- C++ -*-
// 
// Package:     SiPixelPhase1TrackClusters
// Class  :     SiPixelPhase1TrackClusters
//

// Original Author: Marcel Schneider

#include "DQM/SiPixelPhase1Common/interface/SiPixelPhase1Base.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

class SiPixelPhase1TrackClusters : public SiPixelPhase1Base {
  enum {
    ONTRACK_CHARGE,
    ONTRACK_SIZE,
    ONTRACK_NCLUSTERS,
    ONTRACK_POSITION_B,
    ONTRACK_POSITION_F,

    OFFTRACK_CHARGE,
    OFFTRACK_SIZE,
    OFFTRACK_NCLUSTERS,
    OFFTRACK_POSITION_B,
    OFFTRACK_POSITION_F,

    NTRACKS,
    NTRACKS_VOLUME
  };

  public:
  explicit SiPixelPhase1TrackClusters(const edm::ParameterSet& conf);
  void analyze(const edm::Event&, const edm::EventSetup&);

  private:
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > clustersToken_;
  edm::EDGetTokenT<TrajTrackAssociationCollection> trackAssociationToken_;
};

#endif

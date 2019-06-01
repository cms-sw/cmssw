// -*- C++ -*-
//
// Package:     Tracks
// Class  :     FWItemTrackAccessor
//
// Implementation:
//
// Original Author:  Tom McCauley
//         Created:  Thu Feb 18 15:19:44 EDT 2008
//

#include <cassert>

#include "TClass.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "Fireworks/Core/interface/FWItemRandomAccessor.h"
#include "Fireworks/Core/src/FWItemSingleAccessor.h"

class BeamSpotSingleAccessor : public FWItemSingleAccessor {
public:
  BeamSpotSingleAccessor(const TClass* x) : FWItemSingleAccessor(x) {}
  ~BeamSpotSingleAccessor() override{};
  REGISTER_FWITEMACCESSOR_METHODS();
};

REGISTER_TEMPLATE_FWITEMACCESSOR(FWItemDetSetAccessor<edm::DetSetVector<SiStripDigi> >,
                                 edm::DetSetVector<SiStripDigi>,
                                 "SiStripDigiCollectionAccessor");
REGISTER_TEMPLATE_FWITEMACCESSOR(FWItemDetSetAccessor<edm::DetSetVector<PixelDigi> >,
                                 edm::DetSetVector<PixelDigi>,
                                 "SiPixelDigiCollectionAccessor");
REGISTER_TEMPLATE_FWITEMACCESSOR(FWItemNewDetSetAccessor<edmNew::DetSetVector<SiStripCluster> >,
                                 edmNew::DetSetVector<SiStripCluster>,
                                 "SiStripClusterCollectionNewAccessor");
REGISTER_TEMPLATE_FWITEMACCESSOR(FWItemNewDetSetAccessor<edmNew::DetSetVector<SiPixelCluster> >,
                                 edmNew::DetSetVector<SiPixelCluster>,
                                 "SiPixelClusterCollectionNewAccessor");
REGISTER_TEMPLATE_FWITEMACCESSOR(FWItemNewDetSetAccessor<edmNew::DetSetVector<Phase2TrackerCluster1D> >,
                                 edmNew::DetSetVector<Phase2TrackerCluster1D>,
                                 "Phase2TrackerCluster1DCollectionNewAccessor");

REGISTER_FWITEMACCESSOR(BeamSpotSingleAccessor, reco::BeamSpot, "BeamSpotAccessor");

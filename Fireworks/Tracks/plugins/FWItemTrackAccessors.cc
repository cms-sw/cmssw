// -*- C++ -*-
//
// Package:     Tracks
// Class  :     FWItemTrackAccessor
//
// Implementation:
//
// Original Author:  Tom McCauley
//         Created:  Thu Feb 18 15:19:44 EDT 2008
// $Id: FWItemTrackAccessors.cc,v 1.9 2011/03/23 14:30:05 amraktad Exp $
//

#include <assert.h>

#include "TClass.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "Fireworks/Core/interface/FWItemRandomAccessor.h"
#include "Fireworks/Core/src/FWItemSingleAccessor.h"

class BeamSpotSingleAccessor : public FWItemSingleAccessor {
public:
   BeamSpotSingleAccessor(const TClass* x): FWItemSingleAccessor(x){}
   virtual ~BeamSpotSingleAccessor() {};
   REGISTER_FWITEMACCESSOR_METHODS();
};

REGISTER_TEMPLATE_FWITEMACCESSOR( FWItemDetSetAccessor<edm::DetSetVector<SiStripDigi> >,edm::DetSetVector<SiStripDigi>, "SiStripDigiCollectionAccessor" );
REGISTER_TEMPLATE_FWITEMACCESSOR( FWItemDetSetAccessor<edm::DetSetVector<PixelDigi> >, edm::DetSetVector<PixelDigi>, "SiPixelDigiCollectionAccessor" );
REGISTER_TEMPLATE_FWITEMACCESSOR( FWItemNewDetSetAccessor<edmNew::DetSetVector<SiStripCluster> >, edmNew::DetSetVector<SiStripCluster>, "SiStripClusterCollectionNewAccessor" );
REGISTER_TEMPLATE_FWITEMACCESSOR( FWItemNewDetSetAccessor<edmNew::DetSetVector<SiPixelCluster> >, edmNew::DetSetVector<SiPixelCluster>, "SiPixelClusterCollectionNewAccessor" );

REGISTER_FWITEMACCESSOR(BeamSpotSingleAccessor, reco::BeamSpot, "BeamSpotAccessor");

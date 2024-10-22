// -*- C++ -*-
//
// Package:     RecoTauTag/HLTProducers
// Class  :     TauRegionalPixelSeedTrackingRegionEDProducer
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Tue, 13 Sep 2022 21:56:57 GMT
//

// system include files

// user include files

#include "FWCore/Framework/interface/MakerMacros.h"
#include "TauRegionalPixelSeedGenerator.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionEDProducerT.h"
using TauRegionalPixelSeedTrackingRegionEDProducer = TrackingRegionEDProducerT<TauRegionalPixelSeedGenerator>;
DEFINE_FWK_MODULE(TauRegionalPixelSeedTrackingRegionEDProducer);

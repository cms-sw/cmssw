// -*- C++ -*-
//
// Package:     RecoTauTag/HLTProducers
// Class  :     TrackingRegionsFromBeamSpotAndL2TauEDProducer
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Tue, 13 Sep 2022 22:07:47 GMT
//

// system include files

// user include files

#include "FWCore/Framework/interface/MakerMacros.h"

#include "TrackingRegionsFromBeamSpotAndL2Tau.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionEDProducerT.h"
using TrackingRegionsFromBeamSpotAndL2TauEDProducer = TrackingRegionEDProducerT<TrackingRegionsFromBeamSpotAndL2Tau>;
DEFINE_FWK_MODULE(TrackingRegionsFromBeamSpotAndL2TauEDProducer);

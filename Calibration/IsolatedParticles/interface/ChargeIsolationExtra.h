#ifndef CalibrationIsolatedParticlesChargeIsolationExtra_h
#define CalibrationIsolatedParticlesChargeIsolationExtra_h

// system include files
#include <memory>
#include <cmath>
#include <string>
#include <map>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"

namespace spr{

  double chargeIsolation(const edm::Event& iEvent, const edm::EventSetup& iSetup, CaloNavigator<DetId>& navigator, reco::TrackCollection::const_iterator trkItr, edm::Handle<reco::TrackCollection> trkCollection, const CaloSubdetectorGeometry* gEB, const CaloSubdetectorGeometry* gEE, TrackDetectorAssociator& associator, TrackAssociatorParameters& parameters_, int ieta, int iphi, std::string& theTrackQuality, bool debug=false);

  bool   chargeIsolation(const DetId anyCell, CaloNavigator<DetId>& navigator, int deta, int dphi);

  double chargeIsolationEcal(const edm::Event& iEvent, const edm::EventSetup& iSetup, const DetId& coreDet, reco::TrackCollection::const_iterator trkItr, edm::Handle<reco::TrackCollection> trkCollection, const CaloGeometry* geo, const CaloTopology* caloTopology, TrackDetectorAssociator& associator, TrackAssociatorParameters& parameters_, int ieta, int iphi, std::string& theTrackQuality, bool debug=false);

  double chargeIsolationHcal(const edm::Event& iEvent, const edm::EventSetup& iSetup, reco::TrackCollection::const_iterator trkItr, edm::Handle<reco::TrackCollection> trkCollection, const DetId ClosestCell, const HcalTopology* topology, const CaloSubdetectorGeometry* gHB, TrackDetectorAssociator& associator, TrackAssociatorParameters& parameters_, int ieta, int iphi, std::string& theTrackQuality, bool debug=false);
 
}

#endif

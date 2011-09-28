// -*- C++ -*
/* 
Functions to define isolation with respect to charge particles

Authors:  Seema Sharma, Sunanda Banerjee
Created: August 2009
*/

#ifndef CalibrationIsolatedParticlesChargeIsolation_h
#define CalibrationIsolatedParticlesChargeIsolation_h

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
#include "Calibration/IsolatedParticles/interface/CaloPropagateTrack.h"

namespace spr{
  // Returns the maximum energy of a track within a NxN matrix around the
  // impact of a given track on the ECAL surface. It assumes that all tracks
  // are extrapolated to the ECAL surface and stored in a vectore *vdetIds*
  double chargeIsolationEcal(unsigned int trkIndex, std::vector<spr::propagatedTrackID>& vdetIds, const CaloGeometry* geo, const CaloTopology* caloTopology, int ieta, int iphi, bool debug=false);

  // Returns the maximum energy of a track within a NxN matrix around the
  // impact of a given track on the ECAL surface. It extrapolates all tracks
  // in the collection to the ECAL surface in order to do the tests
  double chargeIsolationEcal(const DetId& coreDet, reco::TrackCollection::const_iterator trkItr, edm::Handle<reco::TrackCollection> trkCollection, const CaloGeometry* geo, const CaloTopology* caloTopology, const MagneticField* bField, int ieta, int iphi, std::string& theTrackQuality, bool debug=false);

  // Returns the maximum energy of a track within a NxN matrix around the
  // impact of a given track on the HCAL surface. It assumes that all tracks
  // are extrapolated to the HCAL surface and stored in a vectore *vdetIds*
  double chargeIsolationHcal(unsigned int trkIndex, std::vector<spr::propagatedTrackID> & vdetIds, const HcalTopology* topology, int ieta, int iphi, bool debug=false);

  // Returns the maximum energy of a track within a NxN matrix around the
  // impact of a given track on the HCAL surface. It extrapolates all tracks
  // in the collection to the HCAL surface in order to do the tests
  double chargeIsolationHcal(reco::TrackCollection::const_iterator trkItr, edm::Handle<reco::TrackCollection> trkCollection, const DetId ClosestCell, const HcalTopology* topology, const CaloSubdetectorGeometry* gHB, const MagneticField* bField, int ieta, int iphi, std::string& theTrackQuality, bool debug=false);

  bool chargeIsolation(const DetId anyCell, std::vector<DetId>& vdets) ;

  // Returns the maximum energy of a track within a cone of radius *dR*
  // around the impact poiunt to the ECAL surface
  double coneChargeIsolation(const edm::Event& iEvent, const edm::EventSetup& iSetup, reco::TrackCollection::const_iterator trkItr, edm::Handle<reco::TrackCollection> trkCollection, TrackDetectorAssociator& associator, TrackAssociatorParameters& parameters_, std::string theTrackQuality, int &nNearTRKs, int &nLayers_maxNearP, int &trkQual_maxNearP, double &maxNearP_goodTrk, const GlobalPoint& hpoint1, const GlobalVector& trackMom, double dR);
  double chargeIsolationCone(unsigned int trkIndex, std::vector<spr::propagatedTrackDirection> & trkDirs, double dR, int & nNearTRKs, bool debug=false);

  int coneChargeIsolation(const GlobalPoint& hpoint1, const GlobalPoint& point2, const GlobalVector& trackMom, double dR);
 
}

#endif

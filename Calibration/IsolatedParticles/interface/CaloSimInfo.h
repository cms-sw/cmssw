// -*- C++ -*
/* 
Functions to give the details of parent track of SimHits.

Authors:  Seema Sharma, Sunanda Banerjee
Created: August 2009
*/

#ifndef CalibrationIsolatedParticlesCaloSimInfo_h
#define CalibrationIsolatedParticlesCaloSimInfo_h

// system include files
#include <memory>
#include <map>
#include <vector>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "RecoCaloTools/Navigation/interface/CaloTowerNavigator.h"
#include "Calibration/IsolatedParticles/interface/MatrixECALDetIds.h"

namespace spr{

  // takes the EcalSimHits and returns a map energy matched to SimTrack, photons, neutral hadrons etc.
  template< typename T>
  std::map<std::string,double> eECALSimInfo(const edm::Event&, const DetId& det, const CaloGeometry* geo, const CaloTopology* caloTopology, edm::Handle<T>& hitsEB, edm::Handle<T>& hitsEE, edm::Handle<edm::SimTrackContainer>& SimTk, edm::Handle<edm::SimVertexContainer>& SimVtx, const reco::Track* pTrack, TrackerHitAssociator& associate, int ieta, int iphi, double timeCut=150, bool debug=false);

  template< typename T>
  std::map<std::string,double> eECALSimInfo(const edm::Event&, const DetId& det, const CaloGeometry* geo, const CaloTopology* caloTopology, edm::Handle<T>& hitsEB, edm::Handle<T>& hitsEE, edm::Handle<edm::SimTrackContainer>& SimTk, edm::Handle<edm::SimVertexContainer>& SimVtx, const reco::Track* pTrack, TrackerHitAssociator& associate, int ietaE, int ietaW, int iphiN, int iphiS, double timeCut=150, bool debug=false);
  
  // takes the HcalSimHits and returns a map energy matched to SimTrack, photons, neutral hadrons etc.
  template <typename T>
  std::map<std::string,double> eHCALSimInfo(const edm::Event&, const HcalTopology* topology, const DetId& det, const CaloGeometry* geo, edm::Handle<T>& hits,edm::Handle<edm::SimTrackContainer>& SimTk, edm::Handle<edm::SimVertexContainer>& SimVtx, const reco::Track* pTrack, TrackerHitAssociator& associate, int ieta, int iphi, double timeCut=150, bool includeHO=false, bool debug=false);

  template <typename T>
  std::map<std::string,double> eHCALSimInfo(const edm::Event&, const HcalTopology* topology, const DetId& det, const CaloGeometry* geo, edm::Handle<T>& hits,edm::Handle<edm::SimTrackContainer>& SimTk, edm::Handle<edm::SimVertexContainer>& SimVtx, const reco::Track* pTrack, TrackerHitAssociator& associate, int ietaE, int ietaW, int iphiN, int iphiS, double timeCut=150, bool includeHO=false, bool debug=false);
  
  template <typename T>
  std::map<std::string,double> eHCALSimInfo(const edm::Event& iEvent, const HcalTopology* topology, const DetId& det, edm::Handle<T>& hits, edm::Handle<edm::SimTrackContainer>& SimTk, edm::Handle<edm::SimVertexContainer>& SimVtx, const reco::Track* pTrack, TrackerHitAssociator& associate, int ieta, int iphi, std::vector<int>& multiplicityVector, bool debug=false);

  // Actual function which does the matching of SimHits to SimTracks using geantTrackId
  template <typename T>
  std::map<std::string,double> eCaloSimInfo(std::vector<DetId> vdets, const CaloGeometry* geo, edm::Handle<T>& hitsEB, edm::Handle<T>& hitsEE,  edm::Handle<edm::SimTrackContainer>& SimTk, edm::Handle<edm::SimVertexContainer>& SimVtx, edm::SimTrackContainer::const_iterator trkInfo, double timeCut=150, bool debug=false);

  template <typename T>
  std::map<std::string,double> eCaloSimInfo(const CaloGeometry* geo, edm::Handle<T>& hits, edm::Handle<edm::SimTrackContainer>& SimTk, edm::Handle<edm::SimVertexContainer>& SimVtx, std::vector< typename T::const_iterator> hit, edm::SimTrackContainer::const_iterator trkInfo, double timeCut=150, bool includeHO=false, bool debug=false);

  // Returns total energy of CaloSimHits which originate from the matching SimTrack
  template <typename T>
  double eCaloSimInfo(const edm::Event&, const CaloGeometry* geo, edm::Handle<T>& hits, edm::Handle<edm::SimTrackContainer>& SimTk, edm::Handle<edm::SimVertexContainer>& SimVtx, const reco::Track* pTrack, TrackerHitAssociator& associate, double timeCut=150, bool includeHO=false, bool debug=false);

  template <typename T>
  double eCaloSimInfo(const edm::Event&, const CaloGeometry* geo, edm::Handle<T>& hitsEB, edm::Handle<T>& hitsEE, edm::Handle<edm::SimTrackContainer>& SimTk, edm::Handle<edm::SimVertexContainer>& SimVtx, const reco::Track* pTrack, TrackerHitAssociator& associate, double timeCut=150, bool debug=false);

  template <typename T>
    std::map<std::string,double> eCaloSimInfo(edm::Handle<T>& hits, edm::Handle<edm::SimTrackContainer>& SimTk, edm::Handle<edm::SimVertexContainer>& SimVtx, std::vector< typename T::const_iterator> hit, edm::SimTrackContainer::const_iterator trkInfo, std::vector<int>& multiplicityVector, bool debug=false);

  double timeOfFlight(DetId id, const CaloGeometry* geo, bool debug=false);
}

#include "Calibration/IsolatedParticles/interface/CaloSimInfo.icc"
#endif

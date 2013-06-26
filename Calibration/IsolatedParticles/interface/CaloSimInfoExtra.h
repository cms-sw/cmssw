#ifndef CalibrationIsolatedParticlesCaloSimInfoExtra_h
#define CalibrationIsolatedParticlesCaloSimInfoExtra_h

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
#include "Calibration/IsolatedParticles/interface/CaloSimInfo.h"

namespace spr{

  struct energyMap {
    energyMap() {pdgId=0;}
    int                                   pdgId;
    std::vector<std::pair<DetId,double> > matched;
    std::vector<std::pair<DetId,double> > gamma;
    std::vector<std::pair<DetId,double> > charged;
    std::vector<std::pair<DetId,double> > neutral;
    std::vector<std::pair<DetId,double> > rest;
    std::vector<std::pair<DetId,double> > all;
  };

  // takes the EcalSimHits and returns a map energy matched to SimTrack, photons, neutral hadrons etc.
  template< typename T>
  std::map<std::string,double> eECALSimInfo(const edm::Event&, CaloNavigator<DetId>& navigator, const CaloGeometry* geo, edm::Handle<T>& hits, edm::Handle<edm::SimTrackContainer>& SimTk, edm::Handle<edm::SimVertexContainer>& SimVtx, const reco::Track* pTrack, TrackerHitAssociator& associate, int ieta, int iphi, double timeCut=150, bool debug=false);

  template< typename T>
  std::map<std::string,double> eECALSimInfoTotal(const edm::Event&, const DetId& det, const CaloGeometry* geo, const CaloTopology* caloTopology, edm::Handle<T>& hitsEB, edm::Handle<T>& hitsEE, edm::Handle<edm::SimTrackContainer>& SimTk, edm::Handle<edm::SimVertexContainer>& SimVtx, const reco::Track* pTrack, TrackerHitAssociator& associate, int ieta, int iphi, int itry=-1, double timeCut=150, bool debug=false);

  template< typename T>
  energyMap eECALSimInfoMatrix(const edm::Event&, const DetId& det, const CaloGeometry* geo, const CaloTopology* caloTopology, edm::Handle<T>& hitsEB, edm::Handle<T>& hitsEE, edm::Handle<edm::SimTrackContainer>& SimTk, edm::Handle<edm::SimVertexContainer>& SimVtx, const reco::Track* pTrack, TrackerHitAssociator& associate, int ieta, int iphi, double timeCut=150, bool debug=false);
  
  // takes the HcalSimHits and returns a map energy matched to SimTrack, photons, neutral hadrons etc.
  template <typename T>
  std::map<std::string,double> eHCALSimInfoTotal(const edm::Event&, const HcalTopology* topology, const DetId& det, const CaloGeometry* geo, edm::Handle<T>& hits,edm::Handle<edm::SimTrackContainer>& SimTk, edm::Handle<edm::SimVertexContainer>& SimVtx, const reco::Track* pTrack, TrackerHitAssociator& associate, int ieta, int iphi, int itry=-1, double timeCut=150, bool includeHO=false, bool debug=false);

  template <typename T>
  energyMap eHCALSimInfoMatrix(const edm::Event&, const HcalTopology* topology, const DetId& det, const CaloGeometry* geo, edm::Handle<T>& hits,edm::Handle<edm::SimTrackContainer>& SimTk, edm::Handle<edm::SimVertexContainer>& SimVtx, const reco::Track* pTrack, TrackerHitAssociator& associate, int ieta, int iphi, double timeCut=150, bool includeHO=false, bool debug=false);

  // Actual function which does the matching of SimHits to SimTracks using geantTrackId
  template <typename T>
  energyMap caloSimInfoMatrix(const CaloGeometry* geo, edm::Handle<T>& hits, edm::Handle<edm::SimTrackContainer>& SimTk, edm::Handle<edm::SimVertexContainer>& SimVtx, std::vector< typename T::const_iterator> hit, edm::SimTrackContainer::const_iterator trkInfo, double timeCut=150, bool includeHO=false, bool debug=false);

  // Functions to study the Hits for which history cannot be traced back 
  template <typename T>
  std::vector<typename T::const_iterator> missedECALHits(const edm::Event&, CaloNavigator<DetId>& navigator, edm::Handle<T>& hits, edm::Handle<edm::SimTrackContainer>& SimTk, edm::Handle<edm::SimVertexContainer>& SimVtx, const reco::Track* pTrack, TrackerHitAssociator& associate, int ieta, int iphi, bool flag, bool debug=false);

  template <typename T>
  std::vector<typename T::const_iterator> missedHCALHits(const edm::Event&, const HcalTopology* topology, const DetId& det, edm::Handle<T>& hits,edm::Handle<edm::SimTrackContainer>& SimTk, edm::Handle<edm::SimVertexContainer>& SimVtx, const reco::Track* pTrack, TrackerHitAssociator& associate, int ieta, int iphi, bool flag, bool includeHO=false, bool debug=false);

  template <typename T>
  std::vector<typename T::const_iterator> missedCaloHits(edm::Handle<T>& hits, std::vector<int> matchedId, std::vector< typename T::const_iterator> caloHits, bool flag, bool includeHO=false, bool debug=false); 
}

#include "Calibration/IsolatedParticles/interface/CaloSimInfoExtra.icc"
#endif

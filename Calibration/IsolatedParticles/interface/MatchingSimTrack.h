#ifndef CalibrationIsolatedParticlesMatchingSimTrack_h
#define CalibrationIsolatedParticlesMatchingSimTrack_h

// system include files
#include <memory>
#include <map>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

//sim track
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"

namespace spr {

  struct simTkInfo {
    simTkInfo() {
      found = false;
      pdgId = 0;
      charge = -99;
    }
    bool found;
    int pdgId;
    double charge;
  };

  //Returns iterator to the SimTrack matching to the given Reco Track
  edm::SimTrackContainer::const_iterator matchedSimTrack(const edm::Event& iEvent,
                                                         edm::Handle<edm::SimTrackContainer>& SimTk,
                                                         edm::Handle<edm::SimVertexContainer>& SimVtx,
                                                         const reco::Track* pTrack,
                                                         TrackerHitAssociator& associate,
                                                         bool debug = false);

  std::vector<int> matchedSimTrackId(const edm::Event&,
                                     edm::Handle<edm::SimTrackContainer>& SimTk,
                                     edm::Handle<edm::SimVertexContainer>& SimVtx,
                                     const reco::Track* pTrack,
                                     TrackerHitAssociator& associate,
                                     bool debug = false);

  simTkInfo matchedSimTrackInfo(unsigned int simTkId,
                                edm::Handle<edm::SimTrackContainer>& SimTk,
                                edm::Handle<edm::SimVertexContainer>& SimVtx,
                                bool debug = false);

  bool validSimTrack(unsigned int simTkId,
                     edm::SimTrackContainer::const_iterator thisTrkItr,
                     edm::Handle<edm::SimTrackContainer>& SimTk,
                     edm::Handle<edm::SimVertexContainer>& SimVtx,
                     bool debug = false);

  //Returns the parent SimTrack of given SimTrack
  edm::SimTrackContainer::const_iterator parentSimTrack(edm::SimTrackContainer::const_iterator thisTrkItr,
                                                        edm::Handle<edm::SimTrackContainer>& SimTk,
                                                        edm::Handle<edm::SimVertexContainer>& SimVtx,
                                                        bool debug = false);
}  // namespace spr

#endif

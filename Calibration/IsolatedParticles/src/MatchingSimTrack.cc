#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "Calibration/IsolatedParticles/interface/MatchingSimTrack.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <sstream>

namespace spr {

  edm::SimTrackContainer::const_iterator matchedSimTrack(const edm::Event& iEvent,
                                                         edm::Handle<edm::SimTrackContainer>& SimTk,
                                                         edm::Handle<edm::SimVertexContainer>& SimVtx,
                                                         const reco::Track* pTrack,
                                                         TrackerHitAssociator& associate,
                                                         bool debug) {
    edm::SimTrackContainer::const_iterator itr = SimTk->end();
    ;

    //Get the vector of PsimHits associated to TrackerRecHits and select the
    //matching SimTrack on the basis of maximum occurance of trackIds
    std::vector<unsigned int> trkId, trkOcc;
    for (auto const& trkHit : pTrack->recHits()) {
      std::vector<PSimHit> matchedSimIds = associate.associateHit(*trkHit);
      for (unsigned int isim = 0; isim < matchedSimIds.size(); isim++) {
        unsigned tkId = matchedSimIds[isim].trackId();
        bool found = false;
        for (unsigned int j = 0; j < trkId.size(); j++) {
          if (tkId == trkId[j]) {
            trkOcc[j]++;
            found = true;
            break;
          }
        }
        if (!found) {
          trkId.push_back(tkId);
          trkOcc.push_back(1);
        }
      }
    }

    if (debug) {
      std::ostringstream st1;
      for (unsigned int isim = 0; isim < trkId.size(); isim++) {
        st1 << "\n trkId " << trkId[isim] << "  Occurance " << trkOcc[isim] << ", ";
      }
      edm::LogVerbatim("IsoTrack") << st1.str();
    }
    int matchedId = 0;

    unsigned int matchSimTrk = 0;
    if (!trkOcc.empty()) {
      unsigned int maxTrkOcc = 0, idxMax = 0;
      for (unsigned int j = 0; j < trkOcc.size(); j++) {
        if (trkOcc[j] > maxTrkOcc) {
          maxTrkOcc = trkOcc[j];
          idxMax = j;
        }
      }
      matchSimTrk = trkId[idxMax];
      for (auto simTrkItr = SimTk->begin(); simTrkItr != SimTk->end(); simTrkItr++) {
        if (simTrkItr->trackId() == matchSimTrk) {
          matchedId = simTrkItr->type();
          if (debug)
            edm::LogVerbatim("IsoTrack") << "matched trackId (maximum occurance) " << matchSimTrk << " type "
                                         << matchedId;
          itr = simTrkItr;
          break;
        }
      }
    }

    if (matchedId == 0 && debug) {
      edm::LogVerbatim("IsoTrack") << "Could not find matched SimTrk and track history now ";
    }
    return itr;
  }

  //Returns a vector of TrackId originating from the matching SimTrack
  std::vector<int> matchedSimTrackId(const edm::Event& iEvent,
                                     edm::Handle<edm::SimTrackContainer>& SimTk,
                                     edm::Handle<edm::SimVertexContainer>& SimVtx,
                                     const reco::Track* pTrack,
                                     TrackerHitAssociator& associate,
                                     bool debug) {
    // get the matching SimTrack
    edm::SimTrackContainer::const_iterator trkInfo =
        spr::matchedSimTrack(iEvent, SimTk, SimVtx, pTrack, associate, debug);
    unsigned int matchSimTrk = trkInfo->trackId();
    if (debug)
      edm::LogVerbatim("IsoTrack") << "matchedSimTrackId finds the SimTrk ID of the current track to be "
                                   << matchSimTrk;
    std::vector<int> matchTkid;
    if (trkInfo->type() != 0) {
      for (auto simTrkItr = SimTk->begin(); simTrkItr != SimTk->end(); simTrkItr++) {
        if (validSimTrack(matchSimTrk, simTrkItr, SimTk, SimVtx, false))
          matchTkid.push_back(static_cast<int>(simTrkItr->trackId()));
      }
    }
    return matchTkid;
  }

  spr::simTkInfo matchedSimTrackInfo(unsigned int simTkId,
                                     edm::Handle<edm::SimTrackContainer>& SimTk,
                                     edm::Handle<edm::SimVertexContainer>& SimVtx,
                                     bool debug) {
    spr::simTkInfo info;
    for (edm::SimTrackContainer::const_iterator simTrkItr = SimTk->begin(); simTrkItr != SimTk->end(); simTrkItr++) {
      if (simTkId == simTrkItr->trackId()) {
        if (spr::validSimTrack(simTkId, simTrkItr, SimTk, SimVtx, debug)) {
          info.found = true;
          info.pdgId = simTrkItr->type();
          info.charge = simTrkItr->charge();
        } else {
          edm::SimTrackContainer::const_iterator parentItr = spr::parentSimTrack(simTrkItr, SimTk, SimVtx, debug);
          if (debug) {
            if (parentItr != SimTk->end())
              edm::LogVerbatim("IsoTrack") << "original parent of " << simTrkItr->trackId() << " "
                                           << parentItr->trackId() << ", " << parentItr->type();
            else
              edm::LogVerbatim("IsoTrack") << "original parent of " << simTrkItr->trackId() << " not found";
          }
          if (parentItr != SimTk->end()) {
            info.found = true;
            info.pdgId = parentItr->type();
            info.charge = parentItr->charge();
          }
        }
        break;
      }
    }
    return info;
  }

  // Checks if this SimTrack=thisTrkItr originates from the one with trackId=simTkId
  bool validSimTrack(unsigned int simTkId,
                     edm::SimTrackContainer::const_iterator thisTrkItr,
                     edm::Handle<edm::SimTrackContainer>& SimTk,
                     edm::Handle<edm::SimVertexContainer>& SimVtx,
                     bool debug) {
    if (debug)
      edm::LogVerbatim("IsoTrack") << "Inside validSimTrack: trackId " << thisTrkItr->trackId() << " vtxIndex "
                                   << thisTrkItr->vertIndex() << " to be matched to " << simTkId;

    //This track originates from simTkId
    if (thisTrkItr->trackId() == simTkId)
      return true;

    //Otherwise trace back the history using SimTracks and SimVertices
    int vertIndex = thisTrkItr->vertIndex();
    if (vertIndex == -1 || vertIndex >= static_cast<int>(SimVtx->size()))
      return false;

    edm::SimVertexContainer::const_iterator simVtxItr = SimVtx->begin();
    for (int iv = 0; iv < vertIndex; iv++)
      simVtxItr++;
    int parent = simVtxItr->parentIndex();
    if (debug)
      edm::LogVerbatim("IsoTrack") << "validSimTrack:: parent index " << parent << " ";

    if (parent < 0 && simVtxItr != SimVtx->begin()) {
      const math::XYZTLorentzVectorD pos1 = simVtxItr->position();
      for (simVtxItr = SimVtx->begin(); simVtxItr != SimVtx->end(); ++simVtxItr) {
        if (simVtxItr->parentIndex() > 0) {
          const math::XYZTLorentzVectorD pos2 = pos1 - simVtxItr->position();
          double dist = pos2.P();
          if (dist < 0.001) {
            parent = simVtxItr->parentIndex();
            break;
          }
        }
      }
    }

    if (debug)
      edm::LogVerbatim("IsoTrack") << "final index " << parent;
    for (edm::SimTrackContainer::const_iterator simTrkItr = SimTk->begin(); simTrkItr != SimTk->end(); simTrkItr++) {
      if (static_cast<int>(simTrkItr->trackId()) == parent && simTrkItr != thisTrkItr)
        return validSimTrack(simTkId, simTrkItr, SimTk, SimVtx, debug);
    }

    return false;
  }

  //Returns the parent of a SimTrack
  edm::SimTrackContainer::const_iterator parentSimTrack(edm::SimTrackContainer::const_iterator thisTrkItr,
                                                        edm::Handle<edm::SimTrackContainer>& SimTk,
                                                        edm::Handle<edm::SimVertexContainer>& SimVtx,
                                                        bool debug) {
    edm::SimTrackContainer::const_iterator itr = SimTk->end();

    int vertIndex = thisTrkItr->vertIndex();
    if (debug)
      edm::LogVerbatim("IsoTrack") << "SimTrackParent " << thisTrkItr->trackId() << " Vertex " << vertIndex << " Type "
                                   << thisTrkItr->type() << " Charge " << static_cast<int>(thisTrkItr->charge());
    if (vertIndex == -1)
      return thisTrkItr;
    else if (vertIndex >= static_cast<int>(SimVtx->size()))
      return itr;

    edm::SimVertexContainer::const_iterator simVtxItr = SimVtx->begin();
    for (int iv = 0; iv < vertIndex; iv++)
      simVtxItr++;
    int parent = simVtxItr->parentIndex();

    if (parent < 0 && simVtxItr != SimVtx->begin()) {
      const math::XYZTLorentzVectorD pos1 = simVtxItr->position();
      for (simVtxItr = SimVtx->begin(); simVtxItr != SimVtx->end(); ++simVtxItr) {
        if (simVtxItr->parentIndex() > 0) {
          const math::XYZTLorentzVectorD pos2 = pos1 - simVtxItr->position();
          double dist = pos2.P();
          if (dist < 0.001) {
            parent = simVtxItr->parentIndex();
            break;
          }
        }
      }
    }
    for (edm::SimTrackContainer::const_iterator simTrkItr = SimTk->begin(); simTrkItr != SimTk->end(); simTrkItr++) {
      if (static_cast<int>(simTrkItr->trackId()) == parent && simTrkItr != thisTrkItr)
        return parentSimTrack(simTrkItr, SimTk, SimVtx, debug);
    }

    return thisTrkItr;
  }

}  // namespace spr

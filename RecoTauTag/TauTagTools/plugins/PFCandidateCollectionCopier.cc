/*
 * PFCandidateCollectionCopier
 *
 * Copy a PFCandidate collection, with the associated track collections
 *
 * Author: Evan K. Friis (UC Davis)
 *
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "DataFormats/Common/interface/OrphanHandle.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/Common/interface/RefToPtr.h"

class PFCandidateCollectionCopier : public edm::EDProducer {
  public:
    PFCandidateCollectionCopier(const edm::ParameterSet& pset);
    virtual ~PFCandidateCollectionCopier(){}
    void produce(edm::Event& evt, const edm::EventSetup& es);
  private:
    edm::InputTag src_;
    // Mapping between old and new refs
    std::map<reco::TrackRef, size_t> trackMap_;
    std::map<reco::GsfTrackRef, size_t> gsfTrackMap_;
    std::map<reco::MuonRef, size_t> muonMap_;

    std::auto_ptr<StringCutObjectSelector<reco::PFCandidate> > cut_;

    bool embedTracks_;
    bool embedGsfTracks_;
    bool embedMuons_;
};

PFCandidateCollectionCopier::PFCandidateCollectionCopier(const edm::ParameterSet& pset) {

  src_ = pset.getParameter<edm::InputTag>("src");

  embedTracks_ = pset.getParameter<bool>("embedTracks");
  embedGsfTracks_ = pset.getParameter<bool>("embedGsfTracks");
  embedMuons_ = pset.getParameter<bool>("embedMuons");

  std::string cut = pset.getParameter<std::string>("cut");
  if (cut != "") {
    cut_.reset(new StringCutObjectSelector<reco::PFCandidate>(cut));
  }

  if (embedTracks_) produces<reco::TrackCollection>("tracks");
  if (embedGsfTracks_) produces<reco::GsfTrackCollection>("gsfTracks");
  if (embedMuons_) produces<reco::MuonCollection>("muons");

  produces<reco::PFCandidateCollection>();
}

void PFCandidateCollectionCopier::produce(edm::Event& evt,
    const edm::EventSetup& es) {

  // Clear the maps
  trackMap_.clear();
  gsfTrackMap_.clear();
  muonMap_.clear();

  // Create output collection
  std::auto_ptr<reco::PFCandidateCollection> output(
      new reco::PFCandidateCollection);

  // Make our embedded output collections
  std::auto_ptr<reco::TrackCollection> tracks(new reco::TrackCollection);
  std::auto_ptr<reco::GsfTrackCollection> gsfTracks(
      new reco::GsfTrackCollection);
  std::auto_ptr<reco::MuonCollection> muons(new reco::MuonCollection);

  edm::Handle<reco::PFCandidateCollection> particleFlow;
  evt.getByLabel(src_, particleFlow);

  // If we aren't cutting, we know how big the collection will be a priori
  if (!cut_.get())
    output->reserve(particleFlow->size());

  // First loop over all the PFCandidates and get the associated tracks
  for(size_t i = 0; i < particleFlow->size(); ++i) {
    reco::PFCandidateRef pfCand(particleFlow, i);

    // Check if we are applying selections to the PFCandidates
    if (cut_.get()) {
      // Skip if it doesn't pass our cut.
      if (!(*cut_)(*pfCand)) {
        continue;
      }
    }

    // Add a copy to our output collection (w/ a source ptr to the original)
    output->push_back(reco::PFCandidate(refToPtr(pfCand)));

    // Check if it has a trackRef
    if (embedTracks_ && pfCand->trackRef().isNonnull()) {
      tracks->push_back(*pfCand->trackRef());
      // The last index in the new track collection corresponds to current ref
      trackMap_[pfCand->trackRef()] = tracks->size()-1;
    }

    // Check if it has a GSF track ref
    if (embedGsfTracks_ && pfCand->gsfTrackRef().isNonnull()) {
      gsfTracks->push_back(*pfCand->gsfTrackRef());
      // The last index in the new track collection corresponds to current ref
      gsfTrackMap_[pfCand->gsfTrackRef()] = gsfTracks->size()-1;
    }

    if (embedMuons_ && pfCand->muonRef().isNonnull()) {
      muons->push_back(*pfCand->muonRef());
      muonMap_[pfCand->muonRef()] = muons->size()-1;
    }
  }


  // Update the different types of refs of our output
  if (embedTracks_) {
    // Put the owned collection into the event
    edm::OrphanHandle<reco::TrackCollection> tracksPut = evt.put(
        tracks, "tracks");

    for (size_t i = 0; i < output->size(); ++i) {
      reco::PFCandidate& cand = output->at(i);
      if (cand.trackRef().isNonnull()) {
        // Find the index in the owned collection
        size_t newCollectionIndex = trackMap_[cand.trackRef()];
        reco::TrackRef newRef(tracksPut, newCollectionIndex);
        // Update the output candidate
        cand.setTrackRef(newRef);
      }
    }
  }

  if (embedGsfTracks_) {
    // Put the owned collection into the event
    edm::OrphanHandle<reco::GsfTrackCollection> gsfTracksPut = evt.put(
        gsfTracks, "gsfTracks");

    for (size_t i = 0; i < output->size(); ++i) {
      reco::PFCandidate& cand = output->at(i);
      if (cand.gsfTrackRef().isNonnull()) {
        // Find the index in the owned collection
        size_t newCollectionIndex = gsfTrackMap_[cand.gsfTrackRef()];
        reco::GsfTrackRef newRef(gsfTracksPut, newCollectionIndex);
        // Update the output candidate
        cand.setGsfTrackRef(newRef);
      }
    }
  }

  if (embedMuons_) {

    // Update the track refs stored by the muons.
    // By construction, muon refs in PFCandidates are always matched to the
    // track.  We have to do this before we put them in the event.
    if (embedTracks_) {
      for (size_t i = 0; i < output->size(); ++i) {
        if (output->at(i).muonRef().isNonnull()) {
          // Get our local copy of the muon
          reco::Muon& muon = muons->at(muonMap_[output->at(i).muonRef()]);
          muon.setTrack(output->at(i).trackRef());
        }
      }
    }

    // Put the owned collection into the event
    edm::OrphanHandle<reco::MuonCollection> muonsPut = evt.put(
        muons, "muons");

    for (size_t i = 0; i < output->size(); ++i) {
      reco::PFCandidate& cand = output->at(i);
      if (cand.muonRef().isNonnull()) {
        // Find the index in the owned collection
        size_t newCollectionIndex = muonMap_[cand.muonRef()];
        reco::MuonRef newRef(muonsPut, newCollectionIndex);
        // Update the output candidate
        cand.setMuonRef(newRef);
      }
    }
  }

  evt.put(output);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFCandidateCollectionCopier);

/**  \class L2MuonCandidateProducer
 * 
 *   Intermediate step in the L2 muons selection.
 *   This class takes the L2 muons (which are reco::Tracks) 
 *   and creates the correspondent reco::RecoChargedCandidate.
 *
 *   Riccardo's comment:
 *   The separation between the L2MuonProducer and this class allows
 *   the interchangeability of the L2MuonProducer and the StandAloneMuonProducer
 *   This class is supposed to be removed once the
 *   L2/STA comparison will be done, then the RecoChargedCandidate
 *   production will be done into the L2MuonProducer class.
 *
 *
 *   \author  J.Alcaraz
 */

#include <string>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

class L2MuonCandidateProducer : public edm::global::EDProducer<> {
public:
  /// constructor with config
  L2MuonCandidateProducer(const edm::ParameterSet&);

  /// destructor
  ~L2MuonCandidateProducer() override;

  /// produce candidates
  void produce(edm::StreamID sid, edm::Event& event, const edm::EventSetup&) const override;

private:
  // StandAlone Collection Label
  edm::InputTag theSACollectionLabel;
  edm::EDGetTokenT<reco::TrackCollection> tracksToken;
};

/// constructor with config
L2MuonCandidateProducer::L2MuonCandidateProducer(const edm::ParameterSet& parameterSet) {
  LogTrace("Muon|RecoMuon|L2MuonCandidateProducer") << " constructor called";

  // StandAlone Collection Label
  theSACollectionLabel = parameterSet.getParameter<edm::InputTag>("InputObjects");
  tracksToken = consumes<reco::TrackCollection>(theSACollectionLabel);
  produces<reco::RecoChargedCandidateCollection>();
}

/// destructor
L2MuonCandidateProducer::~L2MuonCandidateProducer() {
  LogTrace("Muon|RecoMuon|L2MuonCandidateProducer") << " L2MuonCandidateProducer destructor called";
}

/// reconstruct muons
void L2MuonCandidateProducer::produce(edm::StreamID sid, edm::Event& event, const edm::EventSetup& eventSetup) const {
  const std::string metname = "Muon|RecoMuon|L2MuonCandidateProducer";

  // Take the SA container
  LogTrace(metname) << " Taking the StandAlone muons: " << theSACollectionLabel;
  edm::Handle<reco::TrackCollection> tracks;
  event.getByToken(tracksToken, tracks);

  // Create a RecoChargedCandidate collection
  LogTrace(metname) << " Creating the RecoChargedCandidate collection";
  auto candidates = std::make_unique<reco::RecoChargedCandidateCollection>();

  for (unsigned int i = 0; i < tracks->size(); i++) {
    reco::TrackRef tkref(tracks, i);
    reco::Particle::Charge q = tkref->charge();
    reco::Particle::LorentzVector p4(tkref->px(), tkref->py(), tkref->pz(), tkref->p());
    reco::Particle::Point vtx(tkref->vx(), tkref->vy(), tkref->vz());
    int pid = 13;
    if (abs(q) == 1)
      pid = q < 0 ? 13 : -13;
    else
      edm::LogWarning(metname) << "L2MuonCandidate has charge = " << q;
    reco::RecoChargedCandidate cand(q, p4, vtx, pid);
    cand.setTrack(tkref);
    candidates->push_back(cand);
  }

  event.put(std::move(candidates));

  LogTrace(metname) << " Event loaded"
                    << "================================";
}

// declare as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L2MuonCandidateProducer);

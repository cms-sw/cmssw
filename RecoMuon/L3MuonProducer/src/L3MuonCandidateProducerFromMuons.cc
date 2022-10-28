/**  \class L3MuonCandidateProducerFromMuons
 * 
 *   This class takes reco::Muons and creates
 *   the correspondent reco::RecoChargedCandidate.
 *
 */

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "RecoMuon/L3MuonProducer/src/L3MuonCandidateProducerFromMuons.h"

// Input and output collections
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"

#include <string>

using namespace edm;
using namespace std;
using namespace reco;

static const std::string category("Muon|RecoMuon|L3MuonCandidateProducerFromMuons");

/// constructor with config
L3MuonCandidateProducerFromMuons::L3MuonCandidateProducerFromMuons(const ParameterSet& parameterSet)
    : m_L3CollectionLabel(parameterSet.getParameter<InputTag>("InputObjects")),  // standAlone Collection Label
      m_muonToken(consumes(m_L3CollectionLabel)),
      m_displacedReco(parameterSet.getParameter<bool>("DisplacedReconstruction")) {
  LogTrace(category) << " constructor called";
  produces<RecoChargedCandidateCollection>();
}

/// destructor
L3MuonCandidateProducerFromMuons::~L3MuonCandidateProducerFromMuons() {
  LogTrace(category) << " L3MuonCandidateProducerFromMuons destructor called";
}

void L3MuonCandidateProducerFromMuons::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("InputObjects", edm::InputTag("hltL3Muons"));
  desc.add<bool>("DisplacedReconstruction", false);
  descriptions.addWithDefaultLabel(desc);
}

/// reconstruct muons
void L3MuonCandidateProducerFromMuons::produce(StreamID, Event& event, const EventSetup& eventSetup) const {
  // Create a RecoChargedCandidate collection
  LogTrace(category) << " Creating the RecoChargedCandidate collection";
  auto candidates = std::make_unique<RecoChargedCandidateCollection>();

  // Take the L3 container
  LogTrace(category) << " Taking the L3/GLB muons: " << m_L3CollectionLabel.label();
  Handle<reco::MuonCollection> muons;
  event.getByToken(m_muonToken, muons);

  if (not muons.isValid()) {
    LogError(category) << muons.whyFailed()->what();
  } else {
    for (unsigned int i = 0; i < muons->size(); i++) {
      // avoids crashing in case the muon is SA only.
      TrackRef tkref;
      if (m_displacedReco) {
        tkref = (*muons)[i].isGlobalMuon() ? (*muons)[i].globalTrack() : (*muons)[i].muonBestTrack();
      } else {
        tkref = (*muons)[i].innerTrack().isNonnull() ? (*muons)[i].innerTrack() : (*muons)[i].muonBestTrack();
      }
      Particle::Charge q = tkref->charge();
      Particle::LorentzVector p4(tkref->px(), tkref->py(), tkref->pz(), tkref->p());
      Particle::Point vtx(tkref->vx(), tkref->vy(), tkref->vz());

      int pid = 13;
      if (abs(q) == 1)
        pid = q < 0 ? 13 : -13;
      else
        LogWarning(category) << "L3MuonCandidate has charge = " << q;
      RecoChargedCandidate cand(q, p4, vtx, pid);

      cand.setTrack(tkref);
      candidates->push_back(cand);
    }
  }
  event.put(std::move(candidates));
}

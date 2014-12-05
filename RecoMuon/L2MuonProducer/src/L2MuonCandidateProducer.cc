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
 *   production will be put into the L2MuonProducer class.
 *
 *
 *   \author  J.Alcaraz
 */

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoMuon/L2MuonProducer/src/L2MuonCandidateProducer.h"

// Input and output collections
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"

#include <string>

using namespace edm;
using namespace std;
using namespace reco;

/// constructor with config
L2MuonCandidateProducer::L2MuonCandidateProducer(const ParameterSet& parameterSet){
  LogTrace("Muon|RecoMuon|L2MuonCandidateProducer")<<" constructor called";

  // StandAlone Collection Label
  theSACollectionLabel = parameterSet.getParameter<InputTag>("InputObjects");
  tracksToken = consumes<reco::TrackCollection>(theSACollectionLabel);
  produces<RecoChargedCandidateCollection>();
}
  
/// destructor
L2MuonCandidateProducer::~L2MuonCandidateProducer(){
  LogTrace("Muon|RecoMuon|L2MuonCandidateProducer")<<" L2MuonCandidateProducer destructor called";
}


/// reconstruct muons
void L2MuonCandidateProducer::produce(edm::StreamID sid, Event& event, const EventSetup& eventSetup) const {
  const string metname = "Muon|RecoMuon|L2MuonCandidateProducer";
  
  // Take the SA container
  LogTrace(metname)<<" Taking the StandAlone muons: "<<theSACollectionLabel;
  Handle<TrackCollection> tracks; 
  event.getByToken(tracksToken,tracks);

  // Create a RecoChargedCandidate collection
  LogTrace(metname)<<" Creating the RecoChargedCandidate collection";
  auto_ptr<RecoChargedCandidateCollection> candidates( new RecoChargedCandidateCollection());

  for (unsigned int i=0; i<tracks->size(); i++) {
      TrackRef tkref(tracks,i);
      Particle::Charge q = tkref->charge();
      Particle::LorentzVector p4(tkref->px(), tkref->py(), tkref->pz(), tkref->p());
      Particle::Point vtx(tkref->vx(),tkref->vy(), tkref->vz());
      int pid = 13;
      if(abs(q)==1) pid = q < 0 ? 13 : -13;
      else LogWarning(metname) << "L2MuonCandidate has charge = "<<q;
      RecoChargedCandidate cand(q, p4, vtx, pid);
      cand.setTrack(tkref);
      candidates->push_back(cand);
  }
  
  event.put(candidates);
 
  LogTrace(metname)<<" Event loaded"
		   <<"================================";
}

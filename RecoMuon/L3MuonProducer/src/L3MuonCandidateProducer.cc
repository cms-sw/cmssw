/**  \class L3MuonCandidateProducer
 * 
 *   Intermediate step in the L3 muons selection.
 *   This class takes the L3 muons (which are reco::Tracks) 
 *   and creates the correspondent reco::RecoChargedCandidate.
 *
 *   Riccardo's comment:
 *   The separation between the L3MuonProducer and this class allows
 *   the interchangeability of the L3MuonProducer and the StandAloneMuonProducer
 *   This class is supposed to be removed once the
 *   L3/STA comparison will be done, then the RecoChargedCandidate
 *   production will be put into the L3MuonProducer class.
 *
 *
 *   \author  J.Alcaraz
 */

// Framework
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoMuon/L3MuonProducer/src/L3MuonCandidateProducer.h"

// Input and output collections
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"

#include <string>

using namespace edm;
using namespace std;
using namespace reco;

/// constructor with config
L3MuonCandidateProducer::L3MuonCandidateProducer(const ParameterSet& parameterSet){
  LogTrace("Muon|RecoMuon|L3MuonCandidateProducer")<<" constructor called";

  // StandAlone Collection Label
  theL3CollectionLabel = parameterSet.getParameter<InputTag>("InputObjects");

  produces<RecoChargedCandidateCollection>();
}
  
/// destructor
L3MuonCandidateProducer::~L3MuonCandidateProducer(){
  LogTrace("Muon|RecoMuon|L3MuonCandidateProducer")<<" L3MuonCandidateProducer destructor called";
}


/// reconstruct muons
void L3MuonCandidateProducer::produce(Event& event, const EventSetup& eventSetup){
  const string metname = "Muon|RecoMuon|L3MuonCandidateProducer";
  
  // Take the L3 container
  LogTrace(metname)<<" Taking the L3/GLB muons: "<<theL3CollectionLabel.label();
  Handle<TrackCollection> tracks; 
  event.getByLabel(theL3CollectionLabel,tracks);

  // Create a RecoChargedCandidate collection
  LogTrace(metname)<<" Creating the RecoChargedCandidate collection";
  auto_ptr<RecoChargedCandidateCollection> candidates( new RecoChargedCandidateCollection());

  for (unsigned int i=0; i<tracks->size(); i++) {
      TrackRef tkref(tracks,i);
      Particle::Charge q = tkref->charge();
      Particle::LorentzVector p4(tkref->px(), tkref->py(), tkref->pz(), tkref->p());
      Particle::Point vtx(tkref->vx(),tkref->vy(), tkref->vz());
      int id = q < 0  ?  13 : -13;
      RecoChargedCandidate cand(q, p4, vtx, id); 
      cand.setTrack(tkref);
      candidates->push_back(cand);
  }
  
  event.put(candidates);
 
  LogTrace(metname)<<" Event loaded"
		   <<"================================";
}

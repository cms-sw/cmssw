/**  \class L3TkMuonProducer
 * 
 *    This module creates a skimed list of reco::Track (pointing to the original TrackExtra and TrackingRecHitOwnedVector
 *    One highest pT track per L1/L2 is selected, requiring some quality.
 *
 *   \author  J-R Vlimant.
 */

// Framework
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoMuon/L3MuonProducer/src/L3TkMuonProducer.h"

// Input and output collections
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeedCollection.h"

#include <string>

using namespace edm;
using namespace std;
using namespace reco;

/// constructor with config
L3TkMuonProducer::L3TkMuonProducer(const ParameterSet& parameterSet){
  LogTrace("Muon|RecoMuon|L3TkMuonProducer")<<" constructor called";

  // StandAlone Collection Label
  theL3CollectionLabel = parameterSet.getParameter<InputTag>("InputObjects");
  produces<TrackCollection>();
}
  
/// destructor
L3TkMuonProducer::~L3TkMuonProducer(){
  LogTrace("Muon|RecoMuon|L3TkMuonProducer")<<" L3TkMuonProducer destructor called";
}


/// reconstruct muons
void L3TkMuonProducer::produce(Event& event, const EventSetup& eventSetup){
  const string metname = "Muon|RecoMuon|L3TkMuonProducer";
  
  // Take the L3 container
  LogTrace(metname)<<" Taking the L3/GLB muons: "<<theL3CollectionLabel.label();
  Handle<TrackCollection> tracks; 
  event.getByLabel(theL3CollectionLabel,tracks);

  //make the LX->L3s pools
  typedef std::pair<uint,uint> pseudoRef;
  typedef std::map<pseudoRef, reco::TrackRef > LXtoL3sMap;
  LXtoL3sMap LXtoL3s;

  uint maxI = tracks->size();
  for (uint i=0;i!=maxI;i++){
    TrackRef tk(tracks,i);
    edm::Ref<L3MuonTrajectorySeedCollection> l3seedRef = tk->seedRef().castTo<edm::Ref<L3MuonTrajectorySeedCollection> >();
    TrackRef l2ref = l3seedRef->l2Track();

    //make a pseudo ref, independantly of L1 or L2
    pseudoRef ref;
    if (l2ref.isNull()){
      l1extra::L1MuonParticleRef l1ref = l3seedRef->l1Particle();
      ref=std::make_pair(l1ref.id().id(),l1ref.key());
    }
    else{
      ref=std::make_pair(l2ref.id().id(),l2ref.key());
    }
   
    //fill the map
    LXtoL3sMap::iterator f=LXtoL3s.find(ref);
    if (f!=LXtoL3s.end()){
      //if already in the map, take the highest pT
      if (f->second->pt() < tk->pt()) f->second=tk;
    }else{
      //insert the pseudo ref
      LXtoL3s[ref] = tk;
    }

  }//end loop of input L3 tracks
    
  //prepare the output
  std::auto_ptr<TrackCollection> outTracks( new TrackCollection(LXtoL3s.size()));
  
  //fill the collection from the map
  LXtoL3sMap::iterator f=LXtoL3s.begin();
  uint i=0;
  for (;f!=LXtoL3s.end();++f){
    //copy the track over
    (*outTracks)[i++] = Track(*(f->second));
  }
  
  //put the collection in the event
  event.put(outTracks);
  
  LogTrace(metname)<<" Event loaded"
		   <<"================================";
}

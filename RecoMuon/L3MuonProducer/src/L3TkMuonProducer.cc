/**  \class L3TkMuonProducer
 * 
 *    This module creates a skimed list of reco::Track (pointing to the original TrackExtra and TrackingRecHitOwnedVector
 *    One highest pT track per L1/L2 is selected, requiring some quality.
 *
 *   \author  J-R Vlimant.
 */

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoMuon/L3MuonProducer/src/L3TkMuonProducer.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"

#include <string>

using namespace edm;
using namespace std;
using namespace reco;

/// constructor with config
L3TkMuonProducer::L3TkMuonProducer(const ParameterSet& parameterSet){
  LogTrace("Muon|RecoMuon|L3TkMuonProducer")<<" constructor called";

  // StandAlone Collection Label
  theL3CollectionLabel = parameterSet.getParameter<InputTag>("InputObjects");
  trackToken_ = consumes<reco::TrackCollection>(theL3CollectionLabel); 
  produces<TrackCollection>();
  produces<TrackExtraCollection>();
  produces<TrackingRecHitCollection>();



  callWhenNewProductsRegistered( [this](const edm::BranchDescription& iBD) {
				   edm::TypeID id(typeid(L3MuonTrajectorySeedCollection));
				   if(iBD.unwrappedTypeID() == id) {
				     this->mayConsume<L3MuonTrajectorySeedCollection>(edm::InputTag{iBD.moduleLabel(), iBD.productInstanceName(),iBD.processName()} );
				   }
				 });



}
  
/// destructor
L3TkMuonProducer::~L3TkMuonProducer(){
  LogTrace("Muon|RecoMuon|L3TkMuonProducer")<<" L3TkMuonProducer destructor called";
}

bool L3TkMuonProducer::sharedSeed(const L3MuonTrajectorySeed& s1,const L3MuonTrajectorySeed& s2){
  //quit right away on nH=0
  if (s1.nHits()==0 || s2.nHits()==0) return false;
  //quit right away if not the same number of hits
  if (s1.nHits()!=s2.nHits()) return false;
  TrajectorySeed::range r1=s1.recHits();
  TrajectorySeed::range r2=s2.recHits();
  TrajectorySeed::const_iterator i1,i2;
  TrajectorySeed::const_iterator & i1_e=r1.second,&i2_e=r2.second;
  TrajectorySeed::const_iterator & i1_b=r1.first,&i2_b=r2.first;
  //quit right away if first detId does not match. front exist because of ==0 ->quit test
  if(i1_b->geographicalId() != i2_b->geographicalId()) return false;
  //then check hit by hit if they are the same
  for (i1=i1_b,i2=i2_b;i1!=i1_e && i2!=i2_e;++i1,++i2){
    if (!i1->sharesInput(&(*i2),TrackingRecHit::all)) return false;
  }
  return true;
}

string printvector(const vector<TrackRef> & v){
  std::stringstream ss;
  for (unsigned int i=0;i!=v.size();++i) {
    if (i!=0) ss<<"\n";
    ss<<"track with ref: "<<v[i].id().id()<<":"<<v[i].key()
      <<" and pT: "<<v[i]->pt()
      <<" with seedRef: "<<v[i]->seedRef().id().id()<<":"<<v[i]->seedRef().key();
  }
  return ss.str();
}

string printvector(const vector<L3TkMuonProducer::SeedRef> & v){
  std::stringstream ss;
  for (unsigned int i=0;i!=v.size();++i){
    if (i!=0) ss<<"\n";
    ss<<"seed ref: "<<v[i].id().id()<<":"<<v[i].key();
    if (v[i]->l2Track().isNull())
      ss<<" and pT: "<<v[i]->l1Particle()->pt()<<" of L1: "<<v[i]->l1Particle().id().id()<<":"<<v[i]->l1Particle().key();
    else 
      ss<<" and pT: "<<v[i]->l2Track()->pt()<<" of L2: "<<v[i]->l2Track().id().id()<<":"<<v[i]->l2Track().key();
  }
  return ss.str();
}

string printseed(const L3TkMuonProducer::SeedRef & s){
  std::stringstream ss;
  ss<<" seed ref: "<<s.id().id()<<":"<<s.key()<<" has "<< s->nHits()<<"rechits";
  TrajectorySeed::range r=s->recHits();
  TrajectorySeed::const_iterator it=r.first;
  for (;it!=r.second;++it)
    ss<<"\n detId: "<<it->geographicalId()<<" position: "<<it->localPosition()<<" and error: "<<it->localPositionError();
  return ss.str();
}

/// reconstruct muons
void L3TkMuonProducer::produce(Event& event, const EventSetup& eventSetup) {
  const string metname = "Muon|RecoMuon|L3TkMuonProducer";
  
  // Take the L3 container
  LogDebug(metname)<<" Taking the L3/GLB muons: "<<theL3CollectionLabel.label();
  Handle<TrackCollection> tracks; 
  event.getByToken(trackToken_,tracks);

  //make the LX->L3s pools
  LXtoL3sMap LXtoL3s;

  unsigned int maxI = tracks->size();
  bool gotL3seeds=false;
  edm::Handle<L3MuonTrajectorySeedCollection> l3seeds;

  //make a list of reference to tracker tracks
  vector<TrackRef> orderedTrackTracks(maxI);
  for (unsigned int i=0;i!=maxI;i++) orderedTrackTracks[i]=TrackRef(tracks,i);
  LogDebug(metname)<<"vector of L3 tracks before ordering:\n"<<printvector(orderedTrackTracks);
  //order them in pT
  sort(orderedTrackTracks.begin(),orderedTrackTracks.end(),trackRefBypT);
  LogDebug(metname)<<"vector of L3 tracks after ordering:\n"<<printvector(orderedTrackTracks);
  //loop over then
  for (unsigned int i=0;i!=maxI;i++) {
    TrackRef & tk=orderedTrackTracks[i];  
    SeedRef l3seedRef = tk->seedRef().castTo<SeedRef>();

    vector<SeedRef> allPossibleOrderedLx; // with identical hit-set
    //add the direct relation
    allPossibleOrderedLx.push_back(l3seedRef);
    LogDebug(metname)<<"adding the seed ref: "<<l3seedRef.id().id()<<":"<<l3seedRef.key()<<" for this tracker track: "<<tk.id().id()<<":"<<tk.key();

    //add the relations due to shared seeds
    //check whether there is a "shared" seed in addition
    if (!gotL3seeds){
      //need to fetch the handle from the ref
      const edm::Provenance & seedsProv=event.getProvenance(l3seedRef.id());
      edm::InputTag l3seedsTag(seedsProv.moduleLabel(), seedsProv.productInstanceName(), seedsProv.processName());
      event.getByLabel(l3seedsTag, l3seeds);
      gotL3seeds=true;
      LogDebug(metname)<<"got seeds handle from: "<<l3seedsTag;
    }
    //loop the other seeds in the collection
    for (unsigned int iS=0;iS!=l3seeds->size();++iS){
      const L3MuonTrajectorySeed & seed = (*l3seeds)[iS];
      const L3MuonTrajectorySeed & thisSeed = *l3seedRef;
      if (l3seedRef.key()==iS) continue; //not care about this one
      //compare this seed with the seed in the collection
      if (sharedSeed(seed,thisSeed)){
	SeedRef thisSharedSeedRef(l3seeds,iS);
	LogDebug(metname)<<"shared seeds: \n"<<printseed(l3seedRef)<<" and: \n"<<printseed(thisSharedSeedRef)
			 <<"\nadding ANOTHER seed ref: "<<thisSharedSeedRef.id().id()<<":"<<thisSharedSeedRef.key()<<" for this tracker track: "<<tk.id().id()<<":"<<tk.key();
	//	edm::LogError(metname)<<" we have a shared seed right there.";
	allPossibleOrderedLx.push_back(thisSharedSeedRef);
      }//seed is shared
    }//loop all other existing seed for overlaps

    //now you have the full list of Lx objects that have seed this tracker track.
    // order the list in pT of Lx objects
    LogDebug(metname)<<"list of possible Lx objects for tracker track: "<<tk.id().id()<<":"<<tk.key()<<" before ordering\n"<<printvector(allPossibleOrderedLx);
    sort(allPossibleOrderedLx.begin(),allPossibleOrderedLx.end(),seedRefBypT);
    LogDebug(metname)<<"list of possible Lx objects for tracker track: "<<tk.id().id()<<":"<<tk.key()<<" after ordering\n"<<printvector(allPossibleOrderedLx);
    // assign this tracker track to the highest pT Lx.
    for (unsigned int iL=0;iL!=allPossibleOrderedLx.size();++iL){
      SeedRef thisRef=allPossibleOrderedLx[iL];
      pseudoRef ref = makePseudoRef(*thisRef);
      LogDebug(metname)<<"seed ref: "<<thisRef.id().id()<<":"<<thisRef.key()<<" transcribe to pseudoref: "<<ref.first<<":"<<ref.second;
      LXtoL3sMap::iterator f=LXtoL3s.find(ref);
      if (f!=LXtoL3s.end()){
	//there's already an entry. because of the prior ordering in pT of the tracker track refs
	// the track ref already there *has* a higher pT: this one cannot compete and should be assigned to the next Lx;
	LogDebug(metname)<<"this tracker track: "<<tk.id().id()<<":"<<tk.key()<<" ("<< tk->pt()<<")"
			 <<"\n cannot compete in pT with track: "<<f->second.first.id().id()<<":"<<f->second.first.key()<<" ("<<f->second.first->pt()<<")"
			 <<"\n already assigned to pseudo ref: "<<ref.first<<":"<<ref.second<<" which corresponds to seedRef: "<<f->second.second.id().id()<<":"<<f->second.second.key();
	continue;
      }else{
	//there was no entry yet. make the assignement
	LogDebug(metname)<<"this tracker track: "<<tk.id().id()<<":"<<tk.key()
			 <<" is assigned to pseudo ref: "<<ref.first<<":"<<ref.second<<" which corresponds to seedRef: "<<thisRef.id().id()<<":"<<thisRef.key();
	LXtoL3s[ref] = std::make_pair(tk,thisRef);
	//once assigned. break
	break;
      }
    }//loop possible Lx for possible assignement
  }//loop over ordered list of tracker track refs


  //prepare the output
  std::auto_ptr<TrackCollection> outTracks( new TrackCollection(LXtoL3s.size()));
  std::auto_ptr<TrackExtraCollection> outTrackExtras( new TrackExtraCollection(LXtoL3s.size()));
  reco::TrackExtraRefProd rTrackExtras = event.getRefBeforePut<TrackExtraCollection>();
  std::auto_ptr<TrackingRecHitCollection> outRecHits( new TrackingRecHitCollection());
  TrackingRecHitRefProd rHits = event.getRefBeforePut<TrackingRecHitCollection>();

  LogDebug(metname)<<"reading the map to make "<< LXtoL3s.size()<<"products.";
  //fill the collection from the map
  LXtoL3sMap::iterator f=LXtoL3s.begin();
  unsigned int i=0;
  for (;f!=LXtoL3s.end();++f,++i){

    LogDebug(metname)<<"copy the track over, and make ref to extra";
    const Track & trk = *(f->second.first);
    (*outTracks)[i] = Track(trk);
    (*outTracks)[i].setExtra( TrackExtraRef(rTrackExtras,i));

    LogDebug(metname)<<"copy the trackExtra too, and change the seedref";
    edm::RefToBase<TrajectorySeed> seedRef(f->second.second);
    //do not use the copy constructor, otherwise the hit Ref are still the same
    (*outTrackExtras)[i] = TrackExtra(
				      trk.outerPosition(), trk.outerMomentum(), trk.outerOk(),
				      trk.innerPosition(), trk.innerMomentum(), trk.innerOk(),
				      trk.outerStateCovariance(), trk.outerDetId(),
				      trk.innerStateCovariance(), trk.innerDetId(),
				      seedRef->direction(),seedRef
				      );

    LogDebug(metname)<<"copy the hits too";
    unsigned int iRH=0;
    for( trackingRecHit_iterator hit = trk.recHitsBegin(); hit != trk.recHitsEnd(); ++ hit,++iRH ) {
      outRecHits->push_back((*hit)->clone());
    }
    (*outTrackExtras)[i].setHits( rHits, 0, iRH);
  }
  
  LogDebug(metname)<<"made: "<<outTracks->size()<<" tracks, "<<outTrackExtras->size()<<" extras and "<<outRecHits->size()<<" rechits.";

  //put the collection in the event
  LogDebug(metname)<<"loading...";
  event.put(outTracks);
  event.put(outTrackExtras);
  event.put(outRecHits);
  LogDebug(metname)<<" Event loaded"
		   <<"================================";
}

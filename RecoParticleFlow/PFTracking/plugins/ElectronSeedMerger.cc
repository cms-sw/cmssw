// -*- C++ -*-
//
// Package:    PFTracking
// Class:      ElectronSeedMerger
// 
// Original Author:  Michele Pioppi

#include "RecoParticleFlow/PFTracking/interface/ElectronSeedMerger.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"  
#include <string>


using namespace edm;
using namespace std;
using namespace reco;

ElectronSeedMerger::ElectronSeedMerger(const ParameterSet& iConfig):
 conf_(iConfig)
{
  LogInfo("ElectronSeedMerger")<<"Electron SeedMerger  started  ";
  

  ecalSeedToken_ = consumes<ElectronSeedCollection>(iConfig.getParameter<InputTag>("EcalBasedSeeds"));
  tkSeedToken_ = consumes<ElectronSeedCollection>(iConfig.getParameter<InputTag>("TkBasedSeeds"));
   
  produces<ElectronSeedCollection>();

}


ElectronSeedMerger::~ElectronSeedMerger()
{
  
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.) 

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
ElectronSeedMerger::produce(Event& iEvent, const EventSetup& iSetup)
{
  //CREATE OUTPUT COLLECTION
  auto_ptr<ElectronSeedCollection> output(new ElectronSeedCollection);

  //HANDLE THE INPUT SEED COLLECTIONS
  Handle<ElectronSeedCollection> EcalBasedSeeds;
  iEvent.getByToken(ecalSeedToken_,EcalBasedSeeds);
  ElectronSeedCollection ESeed = *(EcalBasedSeeds.product());

  Handle<ElectronSeedCollection> TkBasedSeeds;
  iEvent.getByToken(tkSeedToken_,TkBasedSeeds);
  ElectronSeedCollection TSeed = *(TkBasedSeeds.product());


  //VECTOR FOR MATCHED SEEDS
  vector<bool> TSeedMatched;
  for (unsigned int it=0;it<TSeed.size();it++){
    TSeedMatched.push_back(false);
  } 


  //LOOP OVER THE ECAL SEED COLLECTION
  ElectronSeedCollection::const_iterator e_beg= ESeed.begin();
  ElectronSeedCollection::const_iterator e_end= ESeed.end();
  for (;e_beg!=e_end;++e_beg){

    ElectronSeed NewSeed=*(e_beg);
    bool AlreadyMatched =false;
    
    //LOOP OVER THE TK SEED COLLECTION
    for (unsigned int it=0;it<TSeed.size();it++){
      if (AlreadyMatched) continue;

      //HITS FOR ECAL SEED 
      TrajectorySeed::const_iterator eh = e_beg->recHits().first;
      TrajectorySeed::const_iterator eh_end = e_beg->recHits().second;

      //HITS FOR TK SEED 
      unsigned int hitShared=0;
      unsigned int hitSeed=0;
      for (;eh!=eh_end;++eh){

	if (!eh->isValid()) continue;
	hitSeed++;
	bool Shared=false;
	TrajectorySeed::const_iterator th = TSeed[it].recHits().first;
	TrajectorySeed::const_iterator th_end = TSeed[it].recHits().second;
	for (;th!=th_end;++th){
	  if (!th->isValid()) continue;
	  //CHECK THE HIT COMPATIBILITY: put back sharesInput 
	  // as soon Egamma solves the bug on the seed collection
	  if (eh->sharesInput(&(*th),TrackingRecHit::all)) Shared = true;
	//   if(eh->geographicalId() == th->geographicalId() &&
// 	     (eh->localPosition() - th->localPosition()).mag() < 0.001) Shared=true;
	}
	if (Shared) hitShared++;
      }     
      if (hitShared==hitSeed){
	AlreadyMatched=true;
	TSeedMatched[it]=true;
	NewSeed.setCtfTrack(TSeed[it].ctfTrack());
      }
      if ( hitShared == (hitSeed-1)){
	NewSeed.setCtfTrack(TSeed[it].ctfTrack());
      }
    }

    output->push_back(NewSeed);
  }
  
  //FILL THE COLLECTION WITH UNMATCHED TK-BASED SEED
  for (unsigned int it=0;it<TSeed.size();it++){
    if (!TSeedMatched[it])  output->push_back(TSeed[it]);
  }
  
  //PUT THE MERGED COLLECTION IN THE EVENT
  iEvent.put(output);
  
}

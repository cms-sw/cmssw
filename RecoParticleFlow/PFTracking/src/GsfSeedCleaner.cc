// -*- C++ -*-
//
// Package:    PFTracking
// Class:      GsfSeedCleaner
// 
// Original Author:  Michele Pioppi

#include "RecoParticleFlow/PFTracking/interface/GsfSeedCleaner.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

using namespace edm;
using namespace std;
using namespace reco;


GsfSeedCleaner::GsfSeedCleaner(const ParameterSet& iConfig):
  conf_(iConfig)
{
  LogInfo("GsfSeedCleaner")<<"Seed cleaning for Gsf tracks started  ";
  
  //now do what ever initialization is needed
 
  tracksContainers_ = 
    iConfig.getParameter< vector < InputTag > >("TkColList");
 


  preIdLabel_ =
    iConfig.getParameter<InputTag>("PreIdSeedLabel");
 
  produces<TrajectorySeedCollection>();

}



//
// member functions
//

// ------------ method called to produce the data  ------------
void
GsfSeedCleaner::produce(Event& iEvent, const EventSetup& iSetup)
{
  
  LogDebug("GsfSeedCleaner")<<"START event: "<<iEvent.id().event()
			      <<" in run "<<iEvent.id().run();
  
  //Create empty output collections
  auto_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection);

  Handle<TrajectorySeedCollection> SeedCollection;
  iEvent.getByLabel(preIdLabel_,SeedCollection);

  TrajectorySeedCollection::const_iterator isc=SeedCollection->begin();
  TrajectorySeedCollection::const_iterator isc_end=SeedCollection->end();

  for (;isc!=isc_end;++isc){
    bool seed_not_used =true;
    //Vector of track collections
    for (uint istr=0; istr<tracksContainers_.size();istr++){
      
      //Track collection
      Handle<GsfTrackCollection> tkRefCollection;
      iEvent.getByLabel(tracksContainers_[istr], tkRefCollection);


      GsfTrackCollection::const_iterator itk = tkRefCollection->begin();
      GsfTrackCollection::const_iterator itk_end = tkRefCollection->end();
      for(;itk!=itk_end;++itk){

	if (seed_not_used){
	  seed_not_used=CompareHits(*itk,*isc);
	}
      }    
      
    } //end loop on the vector of Gsf track collections
    if (seed_not_used)
      output->push_back(*isc);
  }//end loop on the preid seeds

  iEvent.put(output);
  
}

bool GsfSeedCleaner::CompareHits(const GsfTrack tk,const TrajectorySeed s){


  TrajectorySeed::const_iterator sh = s.recHits().first;
  TrajectorySeed::const_iterator sh_end = s.recHits().second;
  int hitinseed=0;
  int hitshared=0;
  for (;sh!=sh_end;++sh){
    if(!(sh->isValid())) continue;
    hitinseed++;
    trackingRecHit_iterator  ghit= tk.recHitsBegin();
    trackingRecHit_iterator  ghit_end= tk.recHitsEnd();
    for (;ghit!=ghit_end;++ghit){      
      if (!((*ghit)->isValid())) continue;
      	if(((*sh).geographicalId()==(*ghit)->geographicalId())&&
	   (((*sh).localPosition()-(*ghit)->localPosition()).mag()<0.01)) hitshared++;
    }
  }

  if ((hitinseed<3) &&(hitinseed==hitshared)) return false;
  if ((hitinseed>2) &&((hitinseed-hitshared)<2)) return false; 
  return true;
}

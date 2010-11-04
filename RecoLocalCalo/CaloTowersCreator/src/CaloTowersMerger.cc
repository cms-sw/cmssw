// -*- C++ -*-
//
// Package:    CaloTowersMerger
// Class:      CaloTowersMerger
// 
/**\class CaloTowersMerger CaloTowersMerger.cc RecoLocalCalo/CaloTowersMerger/src/CaloTowersMerger.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Jean-Roch Vlimant,40 3-A28,+41227671209,
//         Created:  Thu Nov  4 16:36:30 CET 2010
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoLocalCalo/CaloTowersCreator/src/CaloTowersCreator.h"

//
// class declaration
//

class CaloTowersMerger : public edm::EDProducer {
   public:
      explicit CaloTowersMerger(const edm::ParameterSet&);
      ~CaloTowersMerger();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------

  edm::InputTag towerTag1,towerTag2;
};

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
CaloTowersMerger::CaloTowersMerger(const edm::ParameterSet& iConfig)
{
  towerTag1=iConfig.getParameter<edm::InputTag>("towerTag1");
  towerTag2=iConfig.getParameter<edm::InputTag>("towerTag2");

   //register your products
   produces<CaloTowerCollection>();
}


CaloTowersMerger::~CaloTowersMerger()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
CaloTowersMerger::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<CaloTowerCollection> tower1,tower2;

  iEvent.getByLabel(towerTag1,tower1);
  iEvent.getByLabel(towerTag2,tower2);

  std::auto_ptr<CaloTowerCollection> output;

  if (!tower1.isValid() && !tower2.isValid()){
    edm::LogError("CaloTowersMerger")<<"both input tag:"<<towerTag1<<" and "<<towerTag2<<" are invalid. empty merged collection";
    output.reset(new CaloTowerCollection());
    iEvent.put(output);
    return;
  }else if (!tower1.isValid()  || !tower2.isValid()){
    if (!tower1.isValid() && tower2.isValid())
      tower1=tower2;
    output.reset(new CaloTowerCollection(*tower1));
    iEvent.put(output);
    return;
  }
  else{
    //both valid input collections: merging
    output.reset(new CaloTowerCollection());
    output->reserve(tower1->size()+tower2->size());
  
    //swap to have tower1 with maximum size
    if (tower1->size() < tower2->size()){
      tower1.swap(tower2);
    }
    CaloTowerCollection::const_iterator t1_begin = tower1->begin();
    CaloTowerCollection::const_iterator t1_end = tower1->end();
    CaloTowerCollection::const_iterator t1=t1_begin;

    //vector of overlapping towers
    std::vector<CaloTowerCollection::const_iterator> overlappingTowers;
    overlappingTowers.reserve(tower2->size());

    for (;t1!=t1_end;++t1){
      CaloTowerCollection::const_iterator t2 = tower2->find(t1->id());
      if (t2 != tower2->end()){
	//need to merge the components
	//FIXME
	CaloTower mergedTower(*t1);
	//one needs to merge t1 and t2 into mergedTower
	//end FIXME
	output->push_back(mergedTower);
	overlappingTowers.push_back(t2);
      }else{
	//just copy t1 over
	output->push_back(*t1);
      }
    }
    CaloTowerCollection::const_iterator t2_begin = tower2->begin();
    CaloTowerCollection::const_iterator t2_end = tower2->end();
    CaloTowerCollection::const_iterator t2=t2_begin;
    for (;t2!=t2_end;++t2){
      if (std::find(overlappingTowers.begin(),overlappingTowers.end(),t2)==overlappingTowers.end())
	//non overlapping tower
	//copy t2 over
	output->push_back(*t2);
    }
    iEvent.put(output);
  }
  
}

// ------------ method called once each job just before starting event loop  ------------
void 
CaloTowersMerger::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
CaloTowersMerger::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(CaloTowersMerger);

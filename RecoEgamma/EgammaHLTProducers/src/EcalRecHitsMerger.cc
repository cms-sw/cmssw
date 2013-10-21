
//#include <FWCore/Framework/interface/Handle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "RecoEgamma/EgammaHLTProducers/interface/EcalRecHitsMerger.h"

#include "FWCore/Utilities/interface/Exception.h"


using namespace edm;
using namespace std;


EcalRecHitsMerger::EcalRecHitsMerger(const edm::ParameterSet& pset) {

 debug_ = pset.getUntrackedParameter<bool>("debug");

 EgammaSourceEB_ = pset.getUntrackedParameter<edm::InputTag>("EgammaSource_EB");
 MuonsSourceEB_  = pset.getUntrackedParameter<edm::InputTag>("MuonsSource_EB");
 TausSourceEB_  = pset.getUntrackedParameter<edm::InputTag>("TausSource_EB");
 JetsSourceEB_   = pset.getUntrackedParameter<edm::InputTag>("JetsSource_EB");
 RestSourceEB_   = pset.getUntrackedParameter<edm::InputTag>("RestSource_EB");
 Pi0SourceEB_   = pset.getUntrackedParameter<edm::InputTag>("Pi0Source_EB",edm::InputTag("dummyPi0"));

 EgammaSourceEE_ = pset.getUntrackedParameter<edm::InputTag>("EgammaSource_EE");
 MuonsSourceEE_  = pset.getUntrackedParameter<edm::InputTag>("MuonsSource_EE");
 TausSourceEE_  = pset.getUntrackedParameter<edm::InputTag>("TausSource_EE");
 JetsSourceEE_   = pset.getUntrackedParameter<edm::InputTag>("JetsSource_EE");
 RestSourceEE_   = pset.getUntrackedParameter<edm::InputTag>("RestSource_EE");
 Pi0SourceEE_   = pset.getUntrackedParameter<edm::InputTag>("Pi0Source_EE",edm::InputTag("dummyPi0"));

 OutputLabelEB_ = pset.getUntrackedParameter<std::string>("OutputLabel_EB");
 OutputLabelEE_ = pset.getUntrackedParameter<std::string>("OutputLabel_EE");

 InputRecHitEB_ = pset.getUntrackedParameter<std::string>("EcalRecHitCollectionEB");
 InputRecHitEE_ = pset.getUntrackedParameter<std::string>("EcalRecHitCollectionEE");

 produces<EcalRecHitCollection>(OutputLabelEB_);
 produces<EcalRecHitCollection>(OutputLabelEE_);

}



EcalRecHitsMerger::~EcalRecHitsMerger() {
}

void EcalRecHitsMerger::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {

  edm::ParameterSetDescription desc;
  desc.add<bool>("debug", false);
  desc.add<edm::InputTag>("EgammaSource_EB", edm::InputTag("ecalRegionalEgammaRecHitTmp","EcalRecHitsEB"));
  desc.add<edm::InputTag>("MuonsSource_EB", edm::InputTag("ecalRegionalMuonsRecHitTmp","EcalRecHitsEB"));
  desc.add<edm::InputTag>("TausSource_EB", edm::InputTag("ecalRegionalTausRecHitTmp","EcalRecHitsEB"));
  desc.add<edm::InputTag>("JetsSource_EB", edm::InputTag("ecalRegionalJetsRecHitTmp","EcalRecHitsEB"));
  desc.add<edm::InputTag>("RestSource_EB", edm::InputTag("ecalRegionalRestRecHitTmp","EcalRecHitsEB"));
  desc.add<edm::InputTag>("Pi0Source_EB", edm::InputTag("dummyPi0"));
  desc.add<edm::InputTag>("EgammaSource_EE", edm::InputTag("ecalRegionalEgammaRecHitTmp","EcalRecHitsEE"));
  desc.add<edm::InputTag>("MuonsSource_EE", edm::InputTag("ecalRegionalMuonsRecHitTmp","EcalRecHitsEE"));
  desc.add<edm::InputTag>("TausSource_EE", edm::InputTag("ecalRegionalTausRecHitTmp","EcalRecHitsEE"));
  desc.add<edm::InputTag>("JetsSource_EE", edm::InputTag("ecalRegionalJetsRecHitTmp","EcalRecHitsEE"));
  desc.add<edm::InputTag>("RestSource_EE", edm::InputTag("ecalRegionalRestRecHitTmp","EcalRecHitsEE"));
  desc.add<edm::InputTag>("Pi0Source_EE", edm::InputTag("dummyPi0"));
  desc.add<std::string>("OutputLabel_EB", "EcalRecHitsEB");
  desc.add<std::string>("OutputLabel_EE", "EcalRecHitsEE");
  desc.add<std::string>("EcalRecHitCollectionEB", "EcalRecHitsEB");
  desc.add<std::string>("EcalRecHitCollectionEE", "EcalRecHitsEE");
  descriptions.add("hltEcalRecHitsMerger", desc);  
}

void EcalRecHitsMerger::beginJob(){
}

void EcalRecHitsMerger::endJob(){
}

void EcalRecHitsMerger::produce(edm::Event & e, const edm::EventSetup& iSetup){

 if (debug_) std::cout << " EcalRecHitMerger : Run " << e.id().run() << " Event " << e.id().event() << std::endl;

 std::vector< edm::Handle<EcalRecHitCollection> > EcalRecHits_done;
 e.getManyByType(EcalRecHits_done);

 std::auto_ptr<EcalRecHitCollection> EBMergedRecHits(new EcalRecHitCollection);
 std::auto_ptr<EcalRecHitCollection> EEMergedRecHits(new EcalRecHitCollection);

 unsigned int nColl = EcalRecHits_done.size();

 int nEB = 0;
 int nEE = 0;


 for (unsigned int i=0; i < nColl; i++) {

   std::string instance = EcalRecHits_done[i].provenance()->productInstanceName();
   std::string module_label = EcalRecHits_done[i].provenance()->moduleLabel();

   if ( module_label != EgammaSourceEB_.label() && 
	module_label != MuonsSourceEB_.label() &&
	module_label != JetsSourceEB_.label() &&
 	module_label != TausSourceEB_.label() &&
 	module_label != RestSourceEB_.label() &&
        module_label != Pi0SourceEB_.label() ) continue;

   if (instance == InputRecHitEB_)  {
	nEB += EcalRecHits_done[i] -> size();
   }
   else if (instance == InputRecHitEE_) {
	nEE += EcalRecHits_done[i] -> size();
   }

 }

 EBMergedRecHits -> reserve(nEB);
 EEMergedRecHits -> reserve(nEE);
 if (debug_) std::cout << " Number of EB Rechits to merge  = " << nEB << std::endl;
 if (debug_) std::cout << " Number of EE Rechits to merge  = " << nEE << std::endl;

 for (unsigned int i=0; i < nColl; i++) {
   std::string instance = EcalRecHits_done[i].provenance()->productInstanceName(); 

   std::string module_label = EcalRecHits_done[i].provenance()->moduleLabel();
   if ( module_label != EgammaSourceEB_.label() && 
	module_label != MuonsSourceEB_.label() &&
	module_label != JetsSourceEB_.label() &&
 	module_label != TausSourceEB_.label() &&
 	module_label != RestSourceEB_.label() &&
        module_label != Pi0SourceEB_.label() ) continue;

    if (instance == InputRecHitEB_) {
	for (EcalRecHitCollection::const_iterator it=EcalRecHits_done[i]->begin(); it !=EcalRecHits_done[i]->end(); it++) {
		EBMergedRecHits -> push_back(*it);
  	}
   }
   else if (instance == InputRecHitEE_) {
	for (EcalRecHitCollection::const_iterator it=EcalRecHits_done[i]->begin(); it !=EcalRecHits_done[i]->end(); it++) {
		EEMergedRecHits -> push_back(*it);
	}
   }

 }


 // std::cout << " avant le put " << std::endl;
 e.put(EBMergedRecHits,OutputLabelEB_);
 e.put(EEMergedRecHits,OutputLabelEE_);
 // std::cout << " apres le put " << std::endl;

}


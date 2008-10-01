
//#include <FWCore/Framework/interface/Handle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/ESHandle.h>

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "RecoEgamma/EgammaHLTProducers/interface/EcalRecHitsMergerLowPT.h"

#include "FWCore/Utilities/interface/Exception.h"


using namespace edm;
using namespace std;


EcalRecHitsMergerLowPT::EcalRecHitsMergerLowPT(const edm::ParameterSet& pset) {

 debug_ = pset.getUntrackedParameter<bool>("debug");

 MergedSourceEB_ = pset.getUntrackedParameter<edm::InputTag>("MergedSource_EB");
 LowPTSourceEB_ = pset.getUntrackedParameter<edm::InputTag>("LowPTSource_EB");

 MergedSourceEE_ = pset.getUntrackedParameter<edm::InputTag>("MergedSource_EE");
 LowPTSourceEE_ = pset.getUntrackedParameter<edm::InputTag>("LowPTSource_EE");
 
 
 
 OutputLabelEB_ = pset.getUntrackedParameter<std::string>("OutputLabel_EB");
 OutputLabelEE_ = pset.getUntrackedParameter<std::string>("OutputLabel_EE");

 InputRecHitEB_ = pset.getUntrackedParameter<std::string>("EcalRecHitCollectionEB");
 InputRecHitEE_ = pset.getUntrackedParameter<std::string>("EcalRecHitCollectionEE");
 
 produces<EcalRecHitCollection>(OutputLabelEB_);
 produces<EcalRecHitCollection>(OutputLabelEE_);

}



EcalRecHitsMergerLowPT::~EcalRecHitsMergerLowPT() {
}


void EcalRecHitsMergerLowPT::beginJob(const edm::EventSetup& c){
}

void EcalRecHitsMergerLowPT::endJob(){
}

void EcalRecHitsMergerLowPT::produce(edm::Event & e, const edm::EventSetup& iSetup){

 if (debug_) cout << " EcalRecHitMergerLowPT : Run " << e.id().run() << " Event " << e.id().event() << endl;

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

   //  cout<<" merger_lowPTEG: "<<i<<" "<<instance.c_str()<<" "<<module_label.c_str()<<endl;
   
   
   if ( module_label != MergedSourceEB_.label() && 
	module_label != LowPTSourceEB_.label() 
	
	) continue; 
   

  
   
   if (instance == InputRecHitEB_)  {

     //   cout<<"nRHEB: "<<EcalRecHits_done[i] -> size()<<endl;

	nEB += EcalRecHits_done[i] -> size();
   }
   else if (instance == InputRecHitEE_) {

     //  cout<<"nRHEE: "<<EcalRecHits_done[i] -> size()<<endl;


	nEE += EcalRecHits_done[i] -> size();
   }

 }

 EBMergedRecHits -> reserve(nEB);
 EEMergedRecHits -> reserve(nEE);
 if (debug_) cout << " Number of EB Rechits to merge  = " << nEB << endl;
 if (debug_) cout << " Number of EE Rechits to merge  = " << nEE << endl;

 for (unsigned int i=0; i < nColl; i++) {
   std::string instance = EcalRecHits_done[i].provenance()->productInstanceName(); 

   std::string module_label = EcalRecHits_done[i].provenance()->moduleLabel();
   if ( module_label != MergedSourceEB_.label() && 
	module_label != LowPTSourceEB_.label() 
	
	) continue; 

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


 // cout << " avant le put " << endl;
 e.put(EBMergedRecHits,OutputLabelEB_);
 e.put(EEMergedRecHits,OutputLabelEE_);
 // cout << " apres le put " << endl;

}


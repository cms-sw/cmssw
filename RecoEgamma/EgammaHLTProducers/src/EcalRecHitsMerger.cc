
//#include <FWCore/Framework/interface/Handle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/ESHandle.h>

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "RecoEgamma/EgammaHLTProducers/interface/EcalRecHitsMerger.h"

#include "FWCore/Utilities/interface/Exception.h"


using namespace edm;
using namespace std;


EcalRecHitsMerger::EcalRecHitsMerger(const edm::ParameterSet& pset) {

 debug_ = pset.getUntrackedParameter<bool>("debug");

 EgammaSourceEB_ = pset.getUntrackedParameter<edm::InputTag>("EgammaSource_EB");
 MuonsSourceEB_  = pset.getUntrackedParameter<edm::InputTag>("MuonsSource_EB");
 JetsSourceEB_   = pset.getUntrackedParameter<edm::InputTag>("JetsSource_EB");
 RestSourceEB_   = pset.getUntrackedParameter<edm::InputTag>("RestSource_EB");

 EgammaSourceEE_ = pset.getUntrackedParameter<edm::InputTag>("EgammaSource_EE");
 MuonsSourceEE_  = pset.getUntrackedParameter<edm::InputTag>("MuonsSource_EE");
 JetsSourceEE_   = pset.getUntrackedParameter<edm::InputTag>("JetsSource_EE");
 RestSourceEE_   = pset.getUntrackedParameter<edm::InputTag>("RestSource_EE");

 OutputLabelEB_ = pset.getUntrackedParameter<std::string>("OutputLabel_EB");
 OutputLabelEE_ = pset.getUntrackedParameter<std::string>("OutputLabel_EE");

 produces<EcalRecHitCollection>(OutputLabelEB_);
 produces<EcalRecHitCollection>(OutputLabelEE_);

}



EcalRecHitsMerger::~EcalRecHitsMerger() {
}


void EcalRecHitsMerger::beginJob(const edm::EventSetup& c){
}

void EcalRecHitsMerger::endJob(){
}

void EcalRecHitsMerger::produce(edm::Event & e, const edm::EventSetup& iSetup){

 if (debug_) cout << " EcalRecHitMerger : Run " << e.id().run() << " Event " << e.id().event() << endl;

 Handle<EcalRecHitCollection> EBrechits_egamma;
 Handle<EcalRecHitCollection> EBrechits_muons;
 Handle<EcalRecHitCollection> EBrechits_jets;
 Handle<EcalRecHitCollection> EBrechits_rest;

 Handle<EcalRecHitCollection> EErechits_egamma;
 Handle<EcalRecHitCollection> EErechits_muons;
 Handle<EcalRecHitCollection> EErechits_jets;
 Handle<EcalRecHitCollection> EErechits_rest;

 std::auto_ptr<EcalRecHitCollection> EBMergedRecHits(new EcalRecHitCollection);
 std::auto_ptr<EcalRecHitCollection> EEMergedRecHits(new EcalRecHitCollection);


 try {	
	e.getByLabel(EgammaSourceEB_, EBrechits_egamma);
 }
 catch (...) { ;}

 try {
	e.getByLabel(MuonsSourceEB_,  EBrechits_muons);
 }
 catch (...)  { ; }

 try {
	e.getByLabel(JetsSourceEB_,   EBrechits_jets);
 }
 catch (...) { ;}

 try {
	e.getByLabel(RestSourceEB_,   EBrechits_rest);
 }
 catch (...) { ;}


 try {
        e.getByLabel(EgammaSourceEE_, EErechits_egamma);
 }
 catch (...) { ;}
                                                                                                                               
 try {
        e.getByLabel(MuonsSourceEE_,  EErechits_muons);
 }
 catch (...)  { ; }
                                                                                                                               
 try {
        e.getByLabel(JetsSourceEE_,   EErechits_jets);
 }
 catch (...) { ;}

 try {
	e.getByLabel(RestSourceEE_,   EErechits_rest);
 }
 catch (...)  { ;}


 // EcalRecHitCollection* ptr = MergedRecHits.get();

 int EBN_egamma=0;
 int EBN_muons=0;
 int EBN_jets=0;
 int EBN_rest=0;

 if (EBrechits_egamma.isValid()) EBN_egamma = EBrechits_egamma -> size();
 if (EBrechits_muons.isValid())  EBN_muons  = EBrechits_muons  -> size();
 if (EBrechits_jets.isValid())   EBN_jets   = EBrechits_jets   -> size();
 if (EBrechits_rest.isValid())   EBN_rest   = EBrechits_rest   -> size();

 if (debug_) cout << " EBN_egamma EBN_muons EBN_jets EBN_rest " << EBN_egamma << " " << EBN_muons << " " << EBN_jets << " " << EBN_rest << endl;

 int EBnHits = EBN_egamma+EBN_muons+EBN_jets+EBN_rest;
 EBMergedRecHits -> reserve(EBnHits);

 // int n1=0;

 if (EBN_egamma > 0) {
	if (debug_) cout << " Merging  " << EBN_egamma << " EB RecHits from Egamma. " << endl;
	// EcalRecHitCollection& egamma = (EcalRecHitCollection&)*rechits_egamma;
	// EcalRecHitCollection egamma = *rechits_egamma;
	// ptr -> swap(egamma);
	// n1 += N_egamma;
	for (EcalRecHitCollection::const_iterator it=EBrechits_egamma->begin(); it !=EBrechits_egamma->end(); it++) {
		EBMergedRecHits -> push_back(*it);
	}
 }
 if (EBN_muons > 0) {
	if (debug_) cout << " Merging  " << EBN_muons << " EB RecHits from Muons. " << endl;
	// EcalRecHitCollection* ptr2 = ptr + n1;
	// EcalRecHitCollection& muons = (EcalRecHitCollection&)*rechits_muons;
	// EcalRecHitCollection muons = *rechits_muons;
	// ptr2 -> swap(muons);
	// n1 += N_muons;
	for (EcalRecHitCollection::const_iterator it=EBrechits_muons->begin(); it !=EBrechits_muons->end(); it++) {
		 EBMergedRecHits -> push_back(*it);
	}
 } 
 if (EBN_jets > 0) {
	if (debug_) cout << " Merging  " << EBN_jets << " EB RecHits from Jets. " << endl;
	// EcalRecHitCollection* ptr3 = ptr + n1;
	// EcalRecHitCollection& jets = (EcalRecHitCollection&)*rechits_jets;
	// EcalRecHitCollection jets = *rechits_jets;
	// ptr3 -> swap(jets);
	// copy(rechits_jets->begin(), rechits_jets->end(),MergedRecHits->begin() + n1);
	for (EcalRecHitCollection::const_iterator it=EBrechits_jets->begin(); it != EBrechits_jets->end(); it++) {
		EBMergedRecHits -> push_back(*it);
	}
	// std::swap_ranges(jets.begin(), jets.end(),MergedRecHits->begin() +n1);
 }
 if (EBN_rest > 0) {
	if (debug_) cout << " Merging  " << EBN_rest << " EB RecHits from rest." << endl;
	for (EcalRecHitCollection::const_iterator it=EBrechits_rest->begin(); it != EBrechits_rest->end(); it++) {
		EBMergedRecHits -> push_back(*it);
	}
 }



 int EEN_egamma=0;
 int EEN_muons=0;
 int EEN_jets=0;
 int EEN_rest=0;

 if (EErechits_egamma.isValid()) EEN_egamma = EErechits_egamma -> size();
 if (EErechits_muons.isValid())  EEN_muons  = EErechits_muons  -> size();
 if (EErechits_jets.isValid())   EEN_jets   = EErechits_jets   -> size();
 if (EErechits_rest.isValid())   EEN_rest   = EErechits_rest   -> size();
                                                                                                                               
 if (debug_) cout << " EEN_egamma EEN_muons EEN_jets EEN_rest " << EEN_egamma << " " << EEN_muons << " " << EEN_jets << " " << EEN_rest << endl;
                                                                                                                               
 int EEnHits = EEN_egamma+EEN_muons+EEN_jets + EEN_rest;
 EEMergedRecHits -> reserve(EEnHits);
                                                                                                                               
 // int n1=0;
                                                                                                                               
 if (EEN_egamma > 0) {
        if (debug_) cout << " Merging  " << EEN_egamma << " EE RecHits from Egamma. " << endl;
        for (EcalRecHitCollection::const_iterator it=EErechits_egamma->begin(); it !=EErechits_egamma->end(); it++) {
                EEMergedRecHits -> push_back(*it);
        }
 }
 if (EEN_muons > 0) {
        if (debug_) cout << " Merging  " << EEN_muons << " EE RecHits from Muons. " << endl;
        for (EcalRecHitCollection::const_iterator it=EErechits_muons->begin(); it !=EErechits_muons->end(); it++) {
                 EEMergedRecHits -> push_back(*it);
        }
 }
 if (EEN_jets > 0) {
        if (debug_) cout << " Merging  " << EEN_jets << " EE RecHits from Jets. " << endl;
        for (EcalRecHitCollection::const_iterator it=EErechits_jets->begin(); it != EErechits_jets->end(); it++) {
                EEMergedRecHits -> push_back(*it);
        }
        // std::swap_ranges(jets.begin(), jets.end(),MergedRecHits->begin() +n1);
 }
 if (EEN_rest > 0) {
	if (debug_) cout << " Merging  " << EEN_rest << " EE RecHits from rest ." << endl;
	for (EcalRecHitCollection::const_iterator it=EErechits_rest->begin(); it != EErechits_rest->end(); it++) {
		EEMergedRecHits -> push_back(*it);
	}
 }


 // cout << " avant le put " << endl;
 e.put(EBMergedRecHits,OutputLabelEB_);
 e.put(EEMergedRecHits,OutputLabelEE_);
 // cout << " apres le put " << endl;


}


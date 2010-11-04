// -*- C++ -*-
//
// Package:    WenuCandidateFilter
// Class:      WenuCandidateFilter
// 
/**\class WenuCandidateFilter WenuCandidateFilter.cc EWKSoftware/EDMTupleSkimmerFilter/src/WenuCandidateFilter.cc

 Description: <one line class summary>

 Implementation:
    This class contains a filter that searches the event and finds whether
    it fulfills the W Candidate Criteria. If it fullfills them it creates a
    WenuCandidate and stores it in the event ..............................
    Definition of the Wenu Caldidate: 
    *   event that passes the trigger
    *   with an Gsf electron in fiducial
    *   with ET greater than a (configurable) threshold
    *   matched to an HLT object (configurable) with DR < (configurable)
    *   no second electron  in fiducial (configurable) with ET greater 
        than another (configurable) threshold.
    
    You can keep track of the definition of the Wenu Candidate by searching
    for  or grepping  RETURN (upper case)

 Changes Log:
 12Feb09  First Release of the code for CMSSW_2_2_X
 16Sep09  First Release for CMSSW_3_1_X
 09Dec09  Option to ignore trigger
 23Feb10  Added options to use Conversion Rejection, Expected missing hits
          and valid hit at first PXB
          Added option to calculate these criteria and store them in the pat electron object
          this is done by setting in the configuration the flags
	  calculateValidFirstPXBHit = true
          calculateConversionRejection = true
          calculateExpectedMissinghits = true
          Then the code calculates them and you can access all these from pat::Electron
	  myElec.userInt("PassValidFirstPXBHit")      0 fail, 1 passes
          myElec.userInt("PassConversionRejection")   0 fail, 1 passes
          myElec.userInt("NumberOfExpectedMissingHits") the number of lost hits
 01Mar10  2nd electron option to be isolated if requested by user
 22Jun10  modification to allow for the ICHEP VBTF preselection and the
          creation of the VBTF ntuple
 01Jul10  second electron information added
 Contact:
 Nikolaos.Rompotis@Cern.ch
 Imperial College London

*/


#ifndef WenuCandidateFilter_H
#define WenuCandidateFilter_H
// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
//
#include <vector>
#include <iostream>
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"
//
#include "TString.h"
#include "TMath.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "DataFormats/PatCandidates/interface/TriggerObject.h"
// for conversion finder
#include "RecoEgamma/EgammaTools/interface/ConversionFinder.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
//
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
//
// class declaration
//

class WenuCandidateFilter : public edm::EDFilter {
   public:
      explicit WenuCandidateFilter(const edm::ParameterSet&);
      ~WenuCandidateFilter();

   private:
      virtual Bool_t filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      Bool_t isInFiducial(Double_t eta);
  Bool_t passEleIDCuts(pat::Electron *ele);
      // ----------member data ---------------------------
  Double_t ETCut_;
  Double_t METCut_;
  Double_t ETCut2ndEle_;
  edm::InputTag triggerCollectionTag_;
  edm::InputTag triggerEventTag_;
  std::string hltpath_;
  edm::InputTag hltpathFilter_;
  // quick fix for extra trigger
  Bool_t  useExtraTrigger_;
  std::vector<std::string>   vHltpathExtra_;
  std::vector<edm::InputTag> vHltpathFilterExtra_;
  //
  edm::InputTag electronCollectionTag_;
  edm::InputTag metCollectionTag_;
  edm::InputTag pfMetCollectionTag_;
  edm::InputTag tcMetCollectionTag_;
  edm::InputTag ebRecHits_;
  edm::InputTag eeRecHits_;
  //  edm::InputTag PrimaryVerticesCollection_;
  Double_t BarrelMaxEta_;
  Double_t EndCapMaxEta_;
  Double_t EndCapMinEta_;
  Bool_t useTriggerInfo_;
  Bool_t electronMatched2HLT_;
  Double_t electronMatched2HLT_DR_;
  Bool_t vetoSecondElectronEvents_;
  Bool_t useVetoSecondElectronID_;
  std::string vetoSecondElectronIDType_;
  std::string vetoSecondElectronIDSign_;
  Double_t vetoSecondElectronIDValue_;
  Bool_t useValidFirstPXBHit_;
  Bool_t calculateValidFirstPXBHit_;
  Bool_t useConversionRejection_;
  Double_t dist_, dcot_;
  Bool_t useExpectedMissingHits_;
  Bool_t calculateExpectedMissingHits_;
  Int_t  maxNumberOfExpectedMissingHits_;
  Bool_t useEcalDrivenElectrons_;
  Bool_t useSpikeRejection_;
  Double_t spikeCleaningSwissCrossCut_;
  Bool_t useHLTObjectETCut_;
  Double_t hltObjectETCut_;
  Bool_t storeSecondElectron_;
  //

};
#endif
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
WenuCandidateFilter::WenuCandidateFilter(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
  // *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
  // I N P U T      P A R A M E T E R S  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
  // *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
  // Cuts
  ETCut_ = iConfig.getUntrackedParameter<double>("ETCut");
  METCut_ = iConfig.getUntrackedParameter<double>("METCut");

  vetoSecondElectronEvents_ = iConfig.getUntrackedParameter<bool>("vetoSecondElectronEvents",false);
  storeSecondElectron_ =  iConfig.getUntrackedParameter<Bool_t>("storeSecondElectron");
  if (vetoSecondElectronEvents_ || storeSecondElectron_) {
    ETCut2ndEle_ = iConfig.getUntrackedParameter<double>("ETCut2ndEle");
  }
  vetoSecondElectronIDType_ = iConfig.getUntrackedParameter<std::string>("vetoSecondElectronIDType");
  if (vetoSecondElectronIDType_ != "NULL") {
    useVetoSecondElectronID_ = true;
    vetoSecondElectronIDValue_ = iConfig.getUntrackedParameter<double>("vetoSecondElectronIDValue");
    vetoSecondElectronIDSign_ = iConfig.getUntrackedParameter<std::string>("vetoSecondElectronIDSign","=");
  }
  else useVetoSecondElectronID_ = false;

  useEcalDrivenElectrons_ = iConfig.getUntrackedParameter<Bool_t>("useEcalDrivenElectrons", false);
  //
  // preselection criteria: hit pattern
  useValidFirstPXBHit_ =  iConfig.getUntrackedParameter<Bool_t>("useValidFirstPXBHit",false);
  calculateValidFirstPXBHit_ =  iConfig.getUntrackedParameter<Bool_t>("calculateValidFirstPXBHit",false);
  useConversionRejection_ =   iConfig.getUntrackedParameter<Bool_t>("useConversionRejection",false);
  dist_ = iConfig.getUntrackedParameter<Double_t>("conversionRejectionDist", 0.02);
  dcot_ = iConfig.getUntrackedParameter<Double_t>("conversionRejectionDcot", 0.02);
  useExpectedMissingHits_ = iConfig.getUntrackedParameter<Bool_t>("useExpectedMissingHits",false);
  calculateExpectedMissingHits_ = iConfig.getUntrackedParameter<Bool_t>("calculateExpectedMissingHits",false);
  maxNumberOfExpectedMissingHits_ =  iConfig.getUntrackedParameter<int>("maxNumberOfExpectedMissingHits",1);
  //
  Double_t BarrelMaxEta_D = 1.4442;
  Double_t EndCapMinEta_D = 1.566;
  Double_t EndCapMaxEta_D = 2.5;
  BarrelMaxEta_ = iConfig.getUntrackedParameter<double>("BarrelMaxEta", BarrelMaxEta_D);
  EndCapMaxEta_ = iConfig.getUntrackedParameter<double>("EndCapMaxEta", EndCapMaxEta_D);
  EndCapMinEta_ = iConfig.getUntrackedParameter<double>("EndCapMinEta", EndCapMinEta_D);
  // trigger related
  hltpath_=iConfig.getUntrackedParameter<std::string>("hltpath");
  triggerCollectionTag_=iConfig.getUntrackedParameter<edm::InputTag>("triggerCollectionTag");
  triggerEventTag_=iConfig.getUntrackedParameter<edm::InputTag>("triggerEventTag");
  hltpathFilter_=iConfig.getUntrackedParameter<edm::InputTag>("hltpathFilter");
  useHLTObjectETCut_ = iConfig.getUntrackedParameter<Bool_t>("useHLTObjectETCut", false);
  if (useHLTObjectETCut_) {
    hltObjectETCut_=iConfig.getUntrackedParameter<Double_t>("hltObjectETCut");
  }
  // dirty way to add a second trigger with OR, to be done properly in the next tag
  useExtraTrigger_ = iConfig.getUntrackedParameter<Bool_t>("useExtraTrigger");
  if (useExtraTrigger_) {
    vHltpathExtra_=iConfig.getUntrackedParameter< std::vector< std::string> >("vHltpathExtra");
    vHltpathFilterExtra_=iConfig.getUntrackedParameter< std::vector< edm::InputTag > >("vHltpathFilterExtra");
    if (int(vHltpathExtra_.size()) != int(vHltpathFilterExtra_.size())) {
      std::cout << "WenuCandidateFilter: ERROR IN Configuration: vHltpathExtra and vHltpathFilterExtra"
		<< " should have the same dimensions " << std::endl;
    }
  }
  // trigger matching related:
  useTriggerInfo_ = iConfig.getUntrackedParameter<bool>("useTriggerInfo",true);
  electronMatched2HLT_ = iConfig.getUntrackedParameter<bool>("electronMatched2HLT");
  electronMatched2HLT_DR_=iConfig.getUntrackedParameter<double>("electronMatched2HLT_DR");
  // electrons and other
  electronCollectionTag_=iConfig.getUntrackedParameter<edm::InputTag>("electronCollectionTag");
  //
  metCollectionTag_ = iConfig.getUntrackedParameter<edm::InputTag>("metCollectionTag");
  pfMetCollectionTag_ = iConfig.getUntrackedParameter<edm::InputTag>("pfMetCollectionTag");
  tcMetCollectionTag_ = iConfig.getUntrackedParameter<edm::InputTag>("tcMetCollectionTag");
  //  PrimaryVerticesCollection_ = iConfig.getUntrackedParameter<edm::InputTag>("PrimaryVerticesCollection");
  ebRecHits_ =  iConfig.getUntrackedParameter<edm::InputTag>("ebRecHits");
  eeRecHits_ =  iConfig.getUntrackedParameter<edm::InputTag>("eeRecHits");
  // spike cleaning:
  useSpikeRejection_ = iConfig.getUntrackedParameter<Bool_t>("useSpikeRejection");
  if (useSpikeRejection_) {
    spikeCleaningSwissCrossCut_= iConfig.getUntrackedParameter<Double_t>("spikeCleaningSwissCrossCut");
  }
  //
  // *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
  //
  // now print a summary with what exactly the filter does:
  std::cout << "WenuCandidateFilter: Running Wenu Filter..." << std::endl;
  if (useTriggerInfo_) {
    std::cout << "WenuCandidateFilter: HLT Path   " << hltpath_ << std::endl;
    std::cout << "WenuCandidateFilter: HLT Filter "<<hltpathFilter_<<std::endl;
    if (useExtraTrigger_) {
      for (int itrig=0; itrig < (int) vHltpathExtra_.size(); ++itrig) {
	std::cout << "WenuCandidateFilter: OR " << vHltpathExtra_[itrig] 
		  << " with filter: " << vHltpathFilterExtra_[itrig] << std::endl;
      }
    }
  }
  else {
    std::cout << "WenuCandidateFilter: Trigger info will not be used here" 
	      << std::endl;
  }
  std::cout << "WenuCandidateFilter: ET  > " << ETCut_ << std::endl;
  std::cout << "WenuCandidateFilter: MET > " << METCut_ << std::endl;
  if (vetoSecondElectronEvents_) {
    std::cout << "WenuCandidateFilter: VETO 2nd electron with ET > " 
	      << ETCut2ndEle_ << std::endl;
    if (useVetoSecondElectronID_) {
      std::cout<<"WenuCandidateFilter: VETO 2nd ele ID "  
	       << vetoSecondElectronIDType_ << vetoSecondElectronIDSign_
	       << vetoSecondElectronIDValue_
	       << std::endl;
    }
  }
  else {
    std::cout << "WenuCandidateFilter: No veto for 2nd electron applied " 
	      << std::endl;
  }
  if (storeSecondElectron_) {
    std::cout << "WenuCandidateFilter: Second electron will be stored in the event if it exists."
	      << " Check for second electron existence with myElec.userInt(\"hasSecondElectron\")==1" << std::endl;
  }
  if (useEcalDrivenElectrons_) {
    std::cout << "WenuCandidateFilter: Electron Candidate is required to"
	      << " be ecal driven" << std::endl;
  }
  if (electronMatched2HLT_ && useTriggerInfo_) {
    std::cout << "WenuCandidateFilter: Electron Candidate is required to "
	      << "match an HLT object with DR < " << electronMatched2HLT_DR_
	      << std::endl;
  } else {
    std::cout << "WenuCandidateFilter: Electron Candidate NOT required to "
	      << "match HLT object " << std::endl;
  }
  if (useValidFirstPXBHit_) {
    std::cout << "WenuCandidateFilter: Electron Candidate required to have "
              << "a valid hit in 1st PXB layer " << std::endl;
  }
  if (calculateValidFirstPXBHit_) {
    std::cout << "WenuCandidateFilter: Info about whether there is a valid 1st layer PXB hit "
	      << "will be stored: you can access that later by "
	      << "myElec.userInt(\"PassValidFirstPXBHit\")==1" << std::endl;
  }
  if (useExpectedMissingHits_) {
    std::cout << "WenuCandidateFilter: Electron Candidate required to have "
	      << " less than " << maxNumberOfExpectedMissingHits_ 
	      << " expected hits missing " << std::endl;
    ;
  }
  if (calculateExpectedMissingHits_) {
    std::cout << "WenuCandidateFilter: Missing Hits from expected inner layers "
              << "will be calculated and stored: you can access them later by " 
	      << "myElec.userInt(\"NumberOfExpectedMissingHits\")"   << std::endl;
  }
  if (useConversionRejection_) {
    std::cout << "WenuCandidateFilter: Electron Candidate required to pass "
	      << "EGAMMA Conversion Rejection criteria " << std::endl;
  }
  if (useSpikeRejection_) {
    std::cout << "WenuCandidateFilter: Spike Cleaning will be done with the Swiss Cross Criterion"
	      << " cutting at " << spikeCleaningSwissCrossCut_ << std::endl;
  }
  std::cout << "WenuCandidateFilter: Fiducial Cut: " << std::endl;
  std::cout << "WenuCandidateFilter:    BarrelMax: "<<BarrelMaxEta_<<std::endl;
  std::cout << "WenuCandidateFilter:    EndcapMin: " << EndCapMinEta_
	    << "  EndcapMax: " << EndCapMaxEta_
	    <<std::endl;
  //
  // extra info in the event
  // produces< reco::WenuCandidates > ("selectedWenuCandidates").setBranchAlias
  //  ("selectedWenuCandidates");
  produces< pat::CompositeCandidateCollection > 
    ("selectedWenuCandidates").setBranchAlias("selectedWenuCandidates");

}


WenuCandidateFilter::~WenuCandidateFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
WenuCandidateFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;
   using namespace pat;
   //
   // *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   // TRIGGER REQUIREMENT                       *-*-*-*-*-*-*-*-*-*-*-*-*
   // *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   //                                           *-*-*-*-*-*-*-*-*-*-*-*-*
   // the event should pass the trigger, otherwise no wenu candidate -*-*
   edm::Handle<edm::TriggerResults> HLTResults;
   iEvent.getByLabel(triggerCollectionTag_, HLTResults);
   Int_t passTrigger = 0;  
   if (HLTResults.isValid()) {
     const edm::TriggerNames & triggerNames = iEvent.triggerNames(*HLTResults);
     UInt_t trigger_size = HLTResults->size();
     UInt_t trigger_position = triggerNames.triggerIndex(hltpath_);
     UInt_t trigger_position_extra;
     if (trigger_position < trigger_size)
       passTrigger = (int) HLTResults->accept(trigger_position);
     if (useExtraTrigger_ && passTrigger==0) {
       for (int itrig=0; itrig < (int) vHltpathExtra_.size(); ++itrig) {
	 trigger_position_extra = triggerNames.triggerIndex(vHltpathExtra_[itrig]);
	 if (trigger_position_extra < trigger_size) {
	   passTrigger = (int) HLTResults->accept(trigger_position_extra);
	   if (passTrigger>0)  break;
	 }
       }
     }
   }
   else {
     //std::cout << "TriggerResults missing from this event.." << std::endl;
     if (useTriggerInfo_)
       return false; // RETURN if trigger is missing
   }
   if (passTrigger == 0 && useTriggerInfo_) {
     //std::cout<<"HLT path "<<hltpath_<<" does not fire in this event"
     //     <<std::endl;
     return false; // RETURN if event fails the trigger
   }
   // Int_t numberOfHLTFilterObjects = 0;
   //
   edm::Handle<trigger::TriggerEvent> pHLT;
   iEvent.getByLabel(triggerEventTag_, pHLT);
   const Int_t nF(pHLT->sizeFilters());
   const Int_t filterInd = pHLT->filterIndex(hltpathFilter_);
   std::vector<Int_t> filterIndExtra;
   if (useExtraTrigger_) {
     for (int itrig =0; itrig < (int) vHltpathFilterExtra_.size(); ++itrig) {
       //std::cout << "working on #" << itrig << std::endl;
       //std::cout << "  ---> " << vHltpathFilterExtra_[itrig] << std::endl;
       filterIndExtra.push_back(pHLT->filterIndex(vHltpathFilterExtra_[itrig]));
     }
   }
   bool finalpathfound = false;
   if (nF != filterInd) finalpathfound = true;
   else {
     for (int itrig=0; itrig < (int) filterIndExtra.size(); ++itrig) {
       //std::cout << "working on #" << itrig << std::endl;
       //std::cout << "  ---> " << filterIndExtra[itrig] << std::endl;
       if (nF != filterIndExtra[itrig]) {finalpathfound = true; break;}
     }
   }
   /* this is to find the number of hlt objects if needed
      if (nF != filterInd) {
      const trigger::Vids& VIDS (pHLT->filterIds(filterInd));
      const trigger::Keys& KEYS(pHLT->filterKeys(filterInd));
      const Int_t nI(VIDS.size());
      const Int_t nK(KEYS.size());
      numberOfHLTFilterObjects = (nI>nK)? nI:nK;
      }
   */
   if (not finalpathfound) {
     //std::cout << "HLT Filter " << hltpathFilter_.label() 
     //	       << " was not found in this event..." << std::endl;
     if (useTriggerInfo_)
       return false; // RETURN if event fails the trigger
   }

   const trigger::TriggerObjectCollection& TOC(pHLT->getObjects());
   // *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   // ET CUT: at least one electron in the event with ET>ETCut_-*-*-*-*-*
   // *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   // electron collection
   edm::Handle<pat::ElectronCollection> patElectron;
   iEvent.getByLabel(electronCollectionTag_, patElectron);
   if ( ! patElectron.isValid()) {
     //std::cout << "No electrons found in this event with tag " 
     //	       << electronCollectionTag_  << std::endl;
     return false; // RETURN if no elecs in the event
   }
   const pat::ElectronCollection *pElecs = patElectron.product();
   // MET collection
   edm::Handle<pat::METCollection> patMET;
   iEvent.getByLabel(metCollectionTag_,  patMET);
   edm::Handle<pat::METCollection> patpfMET;
   iEvent.getByLabel(pfMetCollectionTag_,  patpfMET);
   edm::Handle<pat::METCollection> pattcMET;
   iEvent.getByLabel(tcMetCollectionTag_,  pattcMET);
   //
   // Note: best to do Duplicate removal here, since the current
   // implementation does not remove triplicates
   // duplicate removal is on at PAT, but does it remove triplicates?
   //
   //  ------------------------------------------------------------------
   //  Order your electrons: first the ones with the higher ET
   //  ------------------------------------------------------------------
   pat::ElectronCollection::const_iterator elec;
   // check how many electrons there are in the event
   const Int_t Nelecs = pElecs->size();
   if (Nelecs == 0) {
     //std::cout << "No electrons found in this event" << std::endl;
     return false; // RETURN if no elecs in the event
   }
   //
   Int_t  counter = 0;
   std::vector<int> indices;
   std::vector<double> ETs;
   pat::ElectronCollection myElectrons;
   for (elec = pElecs->begin(); elec != pElecs->end(); ++elec) {
     // the definition of  the electron ET is wrt Gsf track eta
     Double_t sc_et=elec->caloEnergy()/TMath::CosH(elec->gsfTrack()->eta());
     indices.push_back(counter); ETs.push_back(sc_et);
     myElectrons.push_back(*elec);
     ++counter;
   }
   const Int_t  event_elec_number = (int) indices.size();
   if (event_elec_number == 0) {
     //std::cout << "No electrons in fiducial were found" << std::endl;
     return false; // RETURN if none of the electron in fiducial
   }
   // be careful now: you allocate some memory with new ...................
   // so whenever you return you must release this memory .................
   Int_t *sorted = new int[event_elec_number];
   Double_t *et = new double[event_elec_number];
   for (Int_t i=0; i<event_elec_number; ++i) {
     et[i] = ETs[i];
   }
   // array sorted now has the indices of the highest ET electrons
   TMath::Sort(event_elec_number, et, sorted, true);
   //
   // if the highest electron in the event has ET < ETCut_ return
   Int_t max_et_index = sorted[0];
   if (ETs[max_et_index] < ETCut_) {
     //std::cout << "Highest ET electron is " << ETs[max_et_index]<<std::endl;
     delete [] sorted;  delete [] et;
     return false; // RETURN if the highest ET elec has ET< ETcut
   }
   //for (Int_t i=0; i< event_elec_number; ++i) {
   //  cout << "elec #"<<i << " " << myElectrons[ sorted[i] ].et() << endl;
   //}
   //
   //
   // get the most high-ET electron:
   pat::Electron maxETelec = myElectrons[max_et_index];

   // now search for a second Gsf electron with ET > ETCut2ndEle_
   pat::Electron secondETelec;
   bool hasSecondElectron = false;
   bool failsSecondElectronCut =  false;
   if (event_elec_number==1) {
     maxETelec.addUserInt("hasSecondElectron",0);
   }
   if (event_elec_number>=2 && vetoSecondElectronEvents_) {
     for (int ie=1; ie<event_elec_number; ++ie) {
       pat::Electron secElec = myElectrons[ sorted[ie] ];
       if (ETs[ sorted[ie] ] > ETCut2ndEle_ && passEleIDCuts(&secElec)) {
	 delete [] sorted;  delete [] et;
	 return false;  // RETURN if you have more that 1 electron with ET>cut
       }
     }
   }
   if (event_elec_number>=2 && storeSecondElectron_) {
     for (int ie=1; ie<event_elec_number; ++ie) {
       pat::Electron secElec = myElectrons[ sorted[ie] ];
       if (ie==1) {
	 maxETelec.addUserInt("hasSecondElectron",1);
	 hasSecondElectron = true;
	 secondETelec = secElec;
       }
       if (ETs[ sorted[ie] ] > ETCut2ndEle_ && passEleIDCuts(&secElec)) {
	 failsSecondElectronCut =  true;
	 maxETelec.addUserInt("failsSecondElectronCut",1);
       }
     }
   }
   if (failsSecondElectronCut == false) maxETelec.addUserInt("failsSecondElectronCut",0);
   //
   // demand that it is in fiducial:
   if (not isInFiducial(maxETelec.caloPosition().eta())) {
       delete [] sorted;  delete [] et;
       return false; // RETURN highest ET electron is not in fiducial
   }
   // demand that it is ecal driven
   if (useEcalDrivenElectrons_) {
     if (not maxETelec.ecalDrivenSeed()) {
       delete [] sorted;  delete [] et;
       return false; // RETURN highest ET electron is not ecal driven
     }
   }
   // spike rejection;
   if (useSpikeRejection_ && maxETelec.isEB()) {
     edm::Handle<EcalRecHitCollection> recHits;
     if (maxETelec.isEB())
       iEvent.getByLabel(ebRecHits_, recHits);
     else 
       iEvent.getByLabel(eeRecHits_, recHits);
     const EcalRecHitCollection *myRecHits = recHits.product();     
     const DetId seedId = maxETelec.superCluster()->seed()->seed();
     EcalSeverityLevelAlgo severity;
     Double_t swissCross = severity.swissCross(seedId, *myRecHits);
     if (swissCross > spikeCleaningSwissCrossCut_) {
       delete [] sorted;  delete [] et;
       return false; // RETURN highest ET electron is a spike
     }
   }
   // NOTICE: all primary vtx information is added in the WenuPlotter now
   // add the primary vtx information in the electron:
   //edm::Handle< std::vector<reco::Vertex> > pVtx;
   //iEvent.getByLabel(PrimaryVerticesCollection_, pVtx);
   //const std::vector<reco::Vertex> Vtx = *(pVtx.product());
   //double pv_x = -999999.;
   //double pv_y = -999999.;
   //double pv_z = -999999.;
   //double ele_tip_pv = -999999.;
   //double ele2nd_tip_pv = -999999.;
   //if (Vtx.size() >=1) {
   //  pv_x = Vtx[0].position().x();
   //  pv_y = Vtx[0].position().y();
   //  pv_z = Vtx[0].position().z();
   //  ele_tip_pv = -maxETelec.gsfTrack()->dxy(Vtx[0].position());
   //  if (hasSecondElectron) {
   //    ele2nd_tip_pv = -secondETelec.gsfTrack()->dxy(Vtx[0].position());
   //  }
   //}
   // extra information about the primary vertex collection
   //std::vector<Int_t> VtxTracksSize;
   //std::vector<Float_t>  VtxNormalizedChi2;
   //for (Int_t i=0; i < (Int_t) Vtx.size(); ++i) {
     // tracks per vtx
   //  Int_t tracksSize = Vtx[i].tracksSize();
   //  VtxTracksSize.push_back(tracksSize);
   //  Float_t nchi2 = Float_t(Vtx[0].normalizedChi2());
   //  VtxNormalizedChi2.push_back(nchi2);
   //}
   //
   //maxETelec.addUserFloat("pv_x", Float_t(pv_x));
   //maxETelec.addUserFloat("pv_x", Float_t(pv_y));
   //maxETelec.addUserFloat("pv_z", Float_t(pv_z));
   //maxETelec.addUserFloat("ele_tip_pv", Float_t(ele_tip_pv));
   //if (hasSecondElectron) {
   //  secondETelec.addUserFloat("ele_tip_pv", Float_t(ele2nd_tip_pv));
   //}
   //std::cout << "** selected ele phi: " << maxETelec.phi()
   //	     << ", eta=" << maxETelec.eta() << ", sihih="
   //	     << maxETelec.scSigmaIEtaIEta() << ", hoe=" 
   // 	     << maxETelec.hadronicOverEm() << ", trackIso: " 
   //	     << maxETelec.trackIso() << ", ecalIso: " << maxETelec.ecalIso()
   //	     << ", hcalIso: " << maxETelec.hcalIso()
   //	     << std::endl;
   //
   // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   // special pre-selection requirements ^^^
   // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   // hit pattern and conversion rejection
   if (useValidFirstPXBHit_ || calculateValidFirstPXBHit_) {
     Bool_t fail = 
       not maxETelec.gsfTrack()->hitPattern().hasValidHitInFirstPixelBarrel();
    if(useValidFirstPXBHit_ && fail) 
      {
	delete [] sorted;  delete [] et;
	//std::cout << "Filter: there is no valid hit in 1st layer PXB" << std::endl;
	return false;
      }
    if (calculateValidFirstPXBHit_) {
      std::string vfpx("PassValidFirstPXBHit");
      if (fail)
	maxETelec.addUserInt(vfpx,0);
      else
      	maxETelec.addUserInt(vfpx,1);
    }
   }
   if (useExpectedMissingHits_ || calculateExpectedMissingHits_) {
     Int_t numberOfInnerHits = (Int_t) maxETelec.gsfTrack()->trackerExpectedHitsInner().numberOfHits();
     if (numberOfInnerHits > maxNumberOfExpectedMissingHits_
	 && useExpectedMissingHits_) {
       delete [] sorted;  delete [] et;
       return false;
     }
     if (calculateExpectedMissingHits_) {
       maxETelec.addUserInt("NumberOfExpectedMissingHits",numberOfInnerHits);
     }
   }
   if (useConversionRejection_) {
     // use of conversion rejection as it is implemented in egamma
       Double_t dist = maxETelec.convDist();
       Double_t dcot = maxETelec.convDcot();
       Bool_t isConv = fabs(dist)<dist_ && fabs(dcot) < dcot_;
       if (isConv) {
       	 delete [] sorted;  delete [] et;
       	 return false;	// RETURN: it is conversion 
       }
   }
   //
   //   std::cout << "HLT matching starts" << std::endl;
   if (electronMatched2HLT_ && useTriggerInfo_) {
     Double_t matched_dr_distance = 999999.;
     Double_t matched_dr_distance_2nd = 999999.;
     Int_t trigger_int_probe = 0;
     if (finalpathfound) {
       if (nF != filterInd) {
	 const trigger::Keys& KEYS(pHLT->filterKeys(filterInd));
	 const Int_t nK(KEYS.size());
	 //std::cout << "Found trig objects #" << nK << std::endl;
	 for (Int_t iTrig = 0;iTrig <nK; ++iTrig ) {
	   const trigger::TriggerObject& TO(TOC[KEYS[iTrig]]);
	   //
	   if (useHLTObjectETCut_) {
	     if (TO.et() < hltObjectETCut_) continue;
	   }
	   Double_t dr_ele_HLT = 
	     reco::deltaR(maxETelec.superCluster()->eta(),maxETelec.superCluster()->phi(),TO.eta(),TO.phi());
	   //std::cout << "-->found dr=" << dr_ele_HLT << std::endl;
	   if (TMath::Abs(dr_ele_HLT) < matched_dr_distance) {
	     matched_dr_distance = dr_ele_HLT;
	   }
	   if (hasSecondElectron) {
	     Double_t dr_ele_HLT_2nd = 
	       reco::deltaR(secondETelec.superCluster()->eta(),secondETelec.superCluster()->phi(),TO.eta(),TO.phi());
	     //std::cout << "-->found dr=" << dr_ele_HLT << std::endl;
	     if (TMath::Abs(dr_ele_HLT_2nd) < matched_dr_distance_2nd) {
	       matched_dr_distance_2nd = dr_ele_HLT_2nd;
	     }
	   }
	 }
       }
       if (useExtraTrigger_) {
	 for (int itrig=0; itrig < (int) filterIndExtra.size(); ++itrig) {
	   if (filterIndExtra[itrig] == nF) continue;
	   //std::cout << "working on #" << itrig << std::endl;
	   //std::cout << "  ---> " << filterIndExtra[itrig] << std::endl;
	   const trigger::Keys& KEYS(pHLT->filterKeys(filterIndExtra[itrig]));
	   const Int_t nK(KEYS.size());
	   //std::cout << "Found trig objects #" << nK << std::endl;
	   for (Int_t iTrig = 0;iTrig <nK; ++iTrig ) {
	     const trigger::TriggerObject& TO(TOC[KEYS[iTrig]]);
	     //
	     Double_t dr_ele_HLT = 
	       reco::deltaR(maxETelec.superCluster()->eta(),maxETelec.superCluster()->phi(),TO.eta(),TO.phi());
	     //std::cout << "-->found dr=" << dr_ele_HLT << std::endl;
	     if (TMath::Abs(dr_ele_HLT) < matched_dr_distance) {
	       matched_dr_distance = dr_ele_HLT;
	     }
	     if (hasSecondElectron) {
	       Double_t dr_ele_HLT_2nd = 
		 reco::deltaR(secondETelec.superCluster()->eta(),secondETelec.superCluster()->phi(),TO.eta(),TO.phi());
	       //std::cout << "-->found dr=" << dr_ele_HLT << std::endl;
	       if (TMath::Abs(dr_ele_HLT_2nd) < matched_dr_distance_2nd) {
		 matched_dr_distance_2nd = dr_ele_HLT_2nd;
	       }
	     }
	   }
	 }
       }
       if (matched_dr_distance < electronMatched2HLT_DR_) {++trigger_int_probe; }
       if (trigger_int_probe == 0) {
	 delete [] sorted;  delete [] et;
	 //std::cout << "Electron could not be matched to an HLT object with "
	 //   << std::endl;
	 return false; // RETURN: electron is not matched to an HLT object
       }
       maxETelec.addUserFloat("HLTMatchingDR", Float_t(matched_dr_distance));
       if (hasSecondElectron) {
	 secondETelec.addUserFloat("HLTMatchingDR", Float_t(matched_dr_distance_2nd));
       }
     }
     else {
       delete [] sorted;  delete [] et;
       //std::cout << "Electron filter not found - should not be like that... "
       // << std::endl;
       return false; // RETURN: electron is not matched to an HLT object
     }
   }
   //   std::cout << "HLT matching has finished" << std::endl;
   // ___________________________________________________________________
   //
   // add information of whether the event passes the following sets of
   // triggers. Currently Hardwired, to be changed in the future
   if (HLTResults.isValid()) {
     
     const  std::string process = triggerCollectionTag_.process();
     //
     std::string  HLTPath[12];
     HLTPath[0] = "HLT_Photon10_L1R"           ;
     HLTPath[1] = "HLT_Photon15_L1R"           ;
     HLTPath[2] = "HLT_Photon20_L1R"           ;
     HLTPath[3] = "HLT_Ele10_LW_L1R"           ;
     HLTPath[4] = "HLT_Ele15_LW_L1R"           ;
     HLTPath[5] = "HLT_Ele20_LW_L1R"           ;
     HLTPath[6] = "HLT_Photon15_Cleaned_L1R"   ;
     HLTPath[7] = "HLT_Ele15_SW_L1R"           ;
     HLTPath[8] = "HLT_Ele15_SW_CaloEleId_L1R" ;
     HLTPath[9] = "HLT_Ele15_SW_EleId_L1R"     ;
     HLTPath[10]= "HLT_Ele17_SW_CaloEleId_L1R" ; 
     HLTPath[11]= "HLT_Ele17_SW_LooseEleId_L1R"; 
     //	      HLT_Ele17_SW_CaloEleId_L1R   HLT_Ele17_SW_LooseEleId_L1R
     edm::InputTag HLTFilterType[12];
     HLTFilterType[0]= edm::InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt10HcalIsolFilter","",process);    // HLT_Photon10_L1R
     HLTFilterType[1]= edm::InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter" ,"",process);   // HLT_Photon15_L1R
     HLTFilterType[2]= edm::InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt20HcalIsolFilter" ,"",process);   // HLT_Photon20_L1R
     HLTFilterType[3]= edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt10PixelMatchFilter","",process); 
     HLTFilterType[4]= edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt15PixelMatchFilter","",process);
     HLTFilterType[5]= edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt15EtFilterESet20","",process);
     HLTFilterType[6]= edm::InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt15CleanedHcalIsolFilter","",process);
     HLTFilterType[7]= edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt15PixelMatchFilter","",process); // HLT_Ele15_SW_L1R
     HLTFilterType[8]= edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt15CaloEleIdPixelMatchFilter","",process);
     HLTFilterType[9]= edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt15EleIdDphiFilter","",process);
     HLTFilterType[10]=edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt17CaloEleIdPixelMatchFilter","",process);
     HLTFilterType[11]=edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt17LEleIdDphiFilter","",process);
     //		      
     Int_t triggerDecision =0;
     UInt_t trigger_size = HLTResults->size();
     for (Int_t i=0; i<12; ++i) {
       const edm::TriggerNames & triggerNames = iEvent.triggerNames(*HLTResults);
       UInt_t trigger_position = triggerNames.triggerIndex(HLTPath[i]);
       Int_t  passTrigger = 0;
       if (trigger_position < trigger_size) {
	 passTrigger = (Int_t) HLTResults->accept(trigger_position);
       }
       if (passTrigger >0) {
	 const Int_t myfilterInd = pHLT->filterIndex(HLTFilterType[i]);
	 if (myfilterInd != nF) {
	   triggerDecision +=  Int_t(TMath::Power(2,i));
	 }
       }
     }
     // add the info in the maxETelec
     maxETelec.addUserInt("triggerDecision",triggerDecision);
   }
   // ___________________________________________________________________
   //
   // get the met now:
   const pat::METCollection *pMet = patMET.product();
   const pat::METCollection::const_iterator met = pMet->begin();
   const pat::MET theMET = *met;
   //
   const pat::METCollection *pPfMet = patpfMET.product();
   const pat::METCollection::const_iterator pfmet = pPfMet->begin();
   const pat::MET thePfMET = *pfmet;
   //
   const pat::METCollection *pTcMet = pattcMET.product();
   const pat::METCollection::const_iterator tcmet = pTcMet->begin();
   const pat::MET theTcMET = *tcmet;

   Double_t metEt = met->et();
   //Double_t metEta = met->eta();
   //Double_t metMt = met->mt();
   //Double_t metPhi = met->phi();
   //Double_t metSig = met->mEtSig();
   //std::cout<<"met properties: et=" << met->et() << ", eta: " <<  met->eta()
   //	     << std::endl;
   // 
   if (metEt < METCut_) {
     delete [] sorted;  delete [] et;
     //std::cout << "MET is " << metEt << std::endl;
     return false;  // RETURN if MET is < Metcut
   }
   //
   //
   // now we have a W candidate ...........................................
   pat::CompositeCandidate wenu;
   wenu.addDaughter(maxETelec, "electron");
   wenu.addDaughter(theMET, "met");
   wenu.addDaughter(thePfMET, "pfmet");
   wenu.addDaughter(theTcMET, "tcmet");
   // the second electron if needed
   if (hasSecondElectron && storeSecondElectron_) {
     wenu.addDaughter(secondETelec, "secondElec");
   }

   auto_ptr<pat::CompositeCandidateCollection> 
     selectedWenuCandidates(new pat::CompositeCandidateCollection);
   selectedWenuCandidates->push_back(wenu);
   //
   iEvent.put( selectedWenuCandidates, "selectedWenuCandidates");

   //
   // release your memory
   delete [] sorted;  delete [] et;
   //
   //std::cout << "Filter accepts this event" << std::endl;
   //
   return true;

}

// ------------ method called once each job just after ending the event loop  -
void 
WenuCandidateFilter::endJob() {
}

Bool_t WenuCandidateFilter::isInFiducial(Double_t eta)
{
  if (TMath::Abs(eta) < BarrelMaxEta_) return true;
  else if (TMath::Abs(eta) < EndCapMaxEta_ && TMath::Abs(eta) > EndCapMinEta_)
    return true;
  return false;

}

Bool_t WenuCandidateFilter::passEleIDCuts(pat::Electron *ele)
{
  if (not useVetoSecondElectronID_)  return true;
  if (not ele->isElectronIDAvailable(vetoSecondElectronIDType_)) {
    std::cout << "WenuCandidateFilter: request ignored: 2nd electron ID type "
	      << "not found in electron object" << std::endl;
    return true;
  }
  if (vetoSecondElectronIDSign_ == ">") {
    if (ele->electronID(vetoSecondElectronIDType_)>vetoSecondElectronIDValue_)
      return true;
    else return false;
  }
  else if (vetoSecondElectronIDSign_ == "<") {
    if (ele->electronID(vetoSecondElectronIDType_)<vetoSecondElectronIDValue_)
      return true;
    else return false;
  }
  else {
    if (TMath::Abs(ele->electronID(vetoSecondElectronIDType_)-
	     vetoSecondElectronIDValue_) < 0.1)
      return true;
    else return false;    
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(WenuCandidateFilter);

// -*- C++ -*-
//
// Package:    ZeeCandidateFilter
// Class:      ZeeCandidateFilter
// 
/**\class ZeeCandidateFilter ZeeCandidateFilter.cc EWKSoftware/EDMTupleSkimmerFilter/src/ZeeCandidateFilter.cc

 Description: <one line class summary>

 Implementation:
    This class contains a filter that searches the event and finds whether
    it fulfills the Z Candidate Criteria. If it fullfills them it creates a
    ZeeCandidate and stores it in the event ..............................
    Definition of the Zee Caldidate: 
    * event that passes the trigger  
    * has 2 Gsf electrons in fiducial 
      with ET greater than a (configurable) threshold
    * at least one of them matched to an HLT Object (configurable)
      with DR < configurable

 Changes Log:
 12Feb09  First Release of the code for CMSSW_2_2_X
 17Sep09  First Release for CMSSW_3_1_X
 09Dec09  Option to ignore trigger
 25Feb10  Added options to use Conversion Rejection, Expected missing hits                      
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
 28May10  Implementation of Spring10 selections
 Contact:
 Nikolaos.Rompotis@Cern.ch
 Imperial College London

*/
//
// Original Author:  Nikolaos Rompotis
//         Created:  Thu Feb 12 11:22:04 CET 2009
// $Id: ZeeCandidateFilter.cc,v 1.6 2010/02/25 15:05:12 rompotis Exp $
//
//

#ifndef ZeeCandidateFilter_H
#define ZeeCandidateFilter_H
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
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "DataFormats/PatCandidates/interface/TriggerObject.h"
//
// for conversion finder 
#include "RecoEgamma/EgammaTools/interface/ConversionFinder.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
//
#include "DataFormats/Scalers/interface/DcsStatus.h"
//
// class declaration
//

class ZeeCandidateFilter : public edm::EDFilter {
   public:
      explicit ZeeCandidateFilter(const edm::ParameterSet&);
      ~ZeeCandidateFilter();

   private:
      virtual Bool_t filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      Bool_t isInFiducial(Double_t eta);
      
      // ----------member data ---------------------------

  edm::InputTag triggerCollectionTag_;
  edm::InputTag triggerEventTag_;
  std::string hltpath_;
  edm::InputTag hltpathFilter_;
  edm::InputTag electronCollectionTag_;
  edm::InputTag metCollectionTag_;
  //
  Double_t BarrelMaxEta_;
  Double_t EndCapMaxEta_;
  Double_t EndCapMinEta_;

  Double_t ETCut_;
  Double_t METCut_;

  Bool_t electronMatched2HLT_;
  Double_t electronMatched2HLT_DR_;
  Bool_t useTriggerInfo_;
  Bool_t calculateValidFirstPXBHit_;
  Bool_t calculateConversionRejection_;
  Double_t dist_, dcot_, dist2_, dcot2_;
  Bool_t calculateExpectedMissingHits_;
  Bool_t dataMagneticFieldSetUp_;
  edm::InputTag dcsTag_;
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
ZeeCandidateFilter::ZeeCandidateFilter(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
  // *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
  // I N P U T      P A R A M E T E R S  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
  // *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
  // Cuts
  Double_t ETCut_D = 20.;
  ETCut_ = iConfig.getUntrackedParameter<double>("ETCut",ETCut_D);
  Double_t METCut_D = 0.;
  METCut_ = iConfig.getUntrackedParameter<double>("METCut",METCut_D);
  //
  //
  Double_t BarrelMaxEta_D = 1.4442;
  Double_t EndCapMinEta_D = 1.56;
  Double_t EndCapMaxEta_D = 2.5;
  BarrelMaxEta_ = iConfig.getUntrackedParameter<double>("BarrelMaxEta",
                                                        BarrelMaxEta_D);
  EndCapMaxEta_ = iConfig.getUntrackedParameter<double>("EndCapMaxEta",
                                                        EndCapMaxEta_D);
  EndCapMinEta_ = iConfig.getUntrackedParameter<double>("EndCapMinEta",
                                                        EndCapMinEta_D);
  // trigger related
  std::string hltpath_D = "HLT_Ele15_LW_L1R";
  edm::InputTag triggerCollectionTag_D("TriggerResults","","HLT");
  edm::InputTag triggerEventTag_D("hltTriggerSummaryAOD", "", "HLT");
  edm::InputTag hltpathFilter_D("hltL1NonIsoHLTNonIsoSingleElectronLWEt15Track\
IsolFilter","","HLT");
  hltpath_=iConfig.getUntrackedParameter<std::string>("hltpath", hltpath_D);
  triggerCollectionTag_=iConfig.getUntrackedParameter<edm::InputTag>
    ("triggerCollectionTag", triggerCollectionTag_D);
  triggerEventTag_=iConfig.getUntrackedParameter<edm::InputTag>
    ("triggerEventTag", triggerEventTag_D);
  hltpathFilter_=iConfig.getUntrackedParameter<edm::InputTag>
    ("hltpathFilter",hltpathFilter_D);
  // electrons and other
  edm::InputTag electronCollectionTag_D("selectedLayer1Electrons","","PAT");
  electronCollectionTag_=iConfig.getUntrackedParameter<edm::InputTag>
    ("electronCollectionTag",electronCollectionTag_D);
  //
  edm::InputTag metCollectionTag_D("selectedLayer1METs","","PAT");
  metCollectionTag_ = iConfig.getUntrackedParameter<edm::InputTag>
    ("metCollectionTag", metCollectionTag_D);
  //
  //
  useTriggerInfo_ = iConfig.getUntrackedParameter<bool>("useTriggerInfo",true);
  electronMatched2HLT_ = iConfig.getUntrackedParameter<bool>("electronMatched2HLT");
  electronMatched2HLT_DR_=iConfig.getUntrackedParameter<double>("electronMatched2HLT_DR");
  //
  // extra preselection variables calculated:
  calculateValidFirstPXBHit_ =
    iConfig.getUntrackedParameter<Bool_t>("calculateValidFirstPXBHit",false);
  calculateConversionRejection_ =
    iConfig.getUntrackedParameter<Bool_t>("calculateConversionRejection",false);
  dist_ = iConfig.getUntrackedParameter<Double_t>("conversionRejectionDist", 0.02);
  dcot_ = iConfig.getUntrackedParameter<Double_t>("conversionRejectionDcot", 0.02);
  dist2_ = iConfig.getUntrackedParameter<Double_t>("conversionRejectionDist2", 0.02);
  dcot2_ = iConfig.getUntrackedParameter<Double_t>("conversionRejectionDcot2", 0.02);
  dataMagneticFieldSetUp_ = iConfig.getUntrackedParameter<Bool_t>("dataMagneticFieldSetUp",false);
  if (dataMagneticFieldSetUp_) {
    dcsTag_ = iConfig.getUntrackedParameter<edm::InputTag>("dcsTag");
  }
  calculateExpectedMissingHits_ =
    iConfig.getUntrackedParameter<Bool_t>("calculateExpectedMissingHits",false);
  //
  // *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
  //
  // now print a summary with what exactly the filter does:
  std::cout << "ZeeCandidateFilter: Running Zee Filter..." << std::endl;
  if (useTriggerInfo_) {
    std::cout << "ZeeCandidateFilter: HLT Path   " << hltpath_ << std::endl;
    std::cout << "ZeeCandidateFilter: HLT Filter " <<hltpathFilter_<<std::endl;
  }
  else {
    std::cout << "ZeeCandidateFilter: Trigger info will not be used here" 
	      << std::endl;
  }
  std::cout << "ZeeCandidateFilter: ET  > " << ETCut_ << std::endl;
  std::cout << "ZeeCandidateFilter: MET > " << METCut_ << std::endl;
  if (electronMatched2HLT_&& useTriggerInfo_) {
    std::cout << "ZeeCandidateFilter: at least one electron is required to "
	      << "match an HLT object with DR < " << electronMatched2HLT_DR_
	      << std::endl;
  } else {
    std::cout << "ZeeCandidateFilter: Electron Candidates NOT required to "
	      << "match HLT object " << std::endl;
  }
  std::cout << "ZeeCandidateFilter: Fiducial Cut: " << std::endl;
  std::cout << "ZeeCandidateFilter:    BarrelMax: "<<BarrelMaxEta_<<std::endl;
  std::cout << "ZeeCandidateFilter:    EndcapMin: " << EndCapMinEta_
	    << "  EndcapMax: " << EndCapMaxEta_
	    <<std::endl;
  if (calculateValidFirstPXBHit_) {
    std::cout << "ZeeCandidateFilter: Info about whether there is a valid 1st layer PXB hit "
              << "will be stored: you can access that later by "
              << "myElec.userInt(\"PassValidFirstPXBHit\")==1" << std::endl;
  }
  if (calculateExpectedMissingHits_) {
    std::cout << "ZeeCandidateFilter: Missing Hits from expected inner layers "
              << "will be calculated and stored: you can access them later by "
              << "myElec.userInt(\"NumberOfExpectedMissingHits\")"   << std::endl;
  }
  if (calculateConversionRejection_) {
    std::cout << "ZeeCandidateFilter: EGAMMA Conversion Rejection criteria "
              << "will be calculated and stored: you can access them later by "
              << "demanding for a successful electron "
              << "myElec.userInt(\"PassConversionRejection\")==1"
              << std::endl;
  }
  if (dataMagneticFieldSetUp_) {
    std::cout << "ZeeCandidateFilter: Data Configuration for Magnetic Field DCS tag " 
	      << dcsTag_  << std::endl;
  }
  // extra info in the event
  produces< pat::CompositeCandidateCollection > ("selectedZeeCandidates").setBranchAlias
    ("selectedZeeCandidates");

}


ZeeCandidateFilter::~ZeeCandidateFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
ZeeCandidateFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;
   using namespace pat;
   // *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   // TRIGGER REQUIREMENT                       *-*-*-*-*-*-*-*-*-*-*-*-*
   // *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   //                                           *-*-*-*-*-*-*-*-*-*-*-*-*
   // the event should pass the trigger, otherwise no zee candidate -*-*
   edm::Handle<edm::TriggerResults> HLTResults;
   iEvent.getByLabel(triggerCollectionTag_, HLTResults);
   Int_t passTrigger = 0;  
   if (HLTResults.isValid()) {
     const edm::TriggerNames & triggerNames = iEvent.triggerNames(*HLTResults);
     UInt_t trigger_size = HLTResults->size();
     UInt_t trigger_position = triggerNames.triggerIndex(hltpath_);
     if (trigger_position < trigger_size)
       passTrigger = (int) HLTResults->accept(trigger_position);
   }
   else {
     //std::cout << "TriggerResults missing from this event.." << std::endl;
     if (useTriggerInfo_)
       return false;  // RETURN: trigger absent in this event
   }
   if (passTrigger == 0 && useTriggerInfo_) {
     //std::cout<<"HLT path "<<hltpath_<<" does not fire in this event"
     //      <<std::endl;
     return false; // RETURN: trigger does not fire
   }
   Int_t numberOfHLTFilterObjects = 0;
   //
   edm::Handle<trigger::TriggerEvent> pHLT;
   iEvent.getByLabel(triggerEventTag_, pHLT);
   const Int_t nF(pHLT->sizeFilters());
   const Int_t filterInd = pHLT->filterIndex(hltpathFilter_);
   if (nF != filterInd) {
     const trigger::Vids& VIDS (pHLT->filterIds(filterInd));
     const trigger::Keys& KEYS(pHLT->filterKeys(filterInd));
     const Int_t nI(VIDS.size());
     const Int_t nK(KEYS.size());
     numberOfHLTFilterObjects = (nI>nK)? nI:nK;
   }
   else {
     //std::cout << "HLT Filter " << hltpathFilter_.label() 
     //	       << " was not found in this event..." << std::endl;
     if (useTriggerInfo_)
       return false; // RETURN: no objects that trigger
   }
   if (numberOfHLTFilterObjects == 0 && useTriggerInfo_) 
     return false; // RETURN: at least 1 HLT object
   // for trigger matching
   const trigger::TriggerObjectCollection& TOC(pHLT->getObjects());
   //
   // *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   // ET CUT: at least one electron in the event with ET>20GeV -*-*-*-*-*
   // *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   // electron collection
   edm::Handle<pat::ElectronCollection> patElectron;
   iEvent.getByLabel(electronCollectionTag_, patElectron);
   if ( ! patElectron.isValid()) {
     //std::cout << "No electrons found in this event..." << std::endl;
     return false; // RETURN: no electrons found in this event
   }
   const pat::ElectronCollection *pElecs = patElectron.product();
   // MET collection
   edm::Handle<pat::METCollection> patMET;
   iEvent.getByLabel(metCollectionTag_, patMET);
   //  ------------------------------------------------------------------
   //  Order your electrons: first the ones with the higher ET
   //  ------------------------------------------------------------------
   pat::ElectronCollection::const_iterator elec;
   // check how many electrons there are in the event
   const Int_t Nelecs = pElecs->size();
   if (Nelecs <= 1) return false; // RETURN: at least 2 electrons in the event
   //
   Int_t  counter = 0;
   std::vector<int> indices;
   std::vector<double> ETs;
   pat::ElectronCollection myElectrons;
   for (elec = pElecs->begin(); elec != pElecs->end(); ++elec) {
     if (isInFiducial(elec->caloPosition().eta())) {
       Double_t sc_et = elec->caloEnergy()/cosh(elec->caloPosition().eta());
       indices.push_back(counter); ETs.push_back(sc_et);
       myElectrons.push_back(*elec);
       ++counter;
     }
   }
   const Int_t  event_elec_number = (int) indices.size();
   //cout << "This event has " << event_elec_number << " electrons" << endl;
   if (event_elec_number <=1) 
     return false; // if none of the electron in fiducial
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
   // if the 2 highest electrons in the event has ET < ETCut_ return
   Int_t max_et_index = sorted[0];
   Int_t max_et_index2 = sorted[1];
   if (ETs[max_et_index]<ETCut_ || ETs[max_et_index2]<ETCut_) {
     delete [] sorted;  delete [] et;
     return false; // RETURN: demand the highest ET electrons ET>ETcut
   }
   //
   // my electrons now:
   pat::Electron maxETelec = myElectrons[max_et_index];
   pat::Electron maxETelec2 = myElectrons[max_et_index2];
   //
   // one of them is matched to an HLT object .......................
   if (electronMatched2HLT_ && useTriggerInfo_) {
     Int_t trigger_int_probe = 0;
     if (nF != filterInd) {
       const trigger::Keys& KEYS(pHLT->filterKeys(filterInd));
       const Int_t nK(KEYS.size());
       for (Int_t iTrig = 0;iTrig <nK; ++iTrig ) {
	 const trigger::TriggerObject& TO(TOC[KEYS[iTrig]]);
	 Double_t dr_ele_HLT = 
	   reco::deltaR(maxETelec.eta(),maxETelec.phi(),TO.eta(),TO.phi());
	 Double_t dr_ele_HLT2 = 
	   reco::deltaR(maxETelec2.eta(),maxETelec2.phi(),TO.eta(),TO.phi());
	 if (TMath::Abs(dr_ele_HLT)<electronMatched2HLT_DR_ ||
	     TMath::Abs(dr_ele_HLT2)<electronMatched2HLT_DR_) {
	   ++trigger_int_probe; break;
	 }
       }
       if (trigger_int_probe == 0) {
	 delete [] sorted;  delete [] et;
	 //std::cout << "Electron could not be matched to an HLT object with "
	 //   << std::endl;
	 return false; // RETURN: electron is not matched to an HLT object
       }
     } else { // you should not get in this unless you have messed up
       delete [] sorted;  delete [] et;
       //std::cout << "Electron filter not found - should not be like that... "
       // << std::endl;
       return false; // RETURN: electron is not matched to an HLT object
     }
   }
   // this is PAT only matching - we have prefered the standard one
   /***************************************************************
   const pat::TriggerObjectStandAloneCollection trigFilter = 
     maxETelec.triggerObjectMatchesByFilter(hltpathFilter_.label());
   const pat::TriggerObjectStandAloneCollection trigFilter2 = 
     maxETelec2.triggerObjectMatchesByFilter(hltpathFilter_.label());

   if ((int) trigFilter.size()==0 && (int) trigFilter2.size()==0) {
     delete [] sorted; delete [] et;
     return false; // RETURN at least one should be HLT matched 
   }
   *************************************************************/
   //
   // special preselection requirement info stored in the electrons
   if (calculateValidFirstPXBHit_) {
     std::string vfpx("PassValidFirstPXBHit");
     Bool_t fail1 =
       not maxETelec.gsfTrack()->hitPattern().hasValidHitInFirstPixelBarrel();
     Bool_t fail2 =
       not maxETelec2.gsfTrack()->hitPattern().hasValidHitInFirstPixelBarrel();
     if (fail1) maxETelec.addUserInt(vfpx,0);
     else maxETelec.addUserInt(vfpx,1);
     if (fail2) maxETelec2.addUserInt(vfpx,0);
     else maxETelec2.addUserInt(vfpx,1);
   }
   if (calculateExpectedMissingHits_) {
     Int_t numberOfLostInnerHits = (int) maxETelec.gsfTrack()->trackerExpectedHitsInner().numberOfHits();
     Int_t numberOfLostInnerHits2 = (int) maxETelec2.gsfTrack()->trackerExpectedHitsInner().numberOfHits();
     maxETelec.addUserInt("NumberOfExpectedMissingHits",numberOfLostInnerHits);
     maxETelec2.addUserInt("NumberOfExpectedMissingHits",numberOfLostInnerHits2);
   }
   if (calculateConversionRejection_) {
     Double_t bfield;
     if (dataMagneticFieldSetUp_) {
       edm::Handle<DcsStatusCollection> dcsHandle;
       iEvent.getByLabel(dcsTag_, dcsHandle);
       // scale factor = 3.801/18166.0 which are
       // average values taken over a stable two
       // week period
       Double_t currentToBFieldScaleFactor = 2.09237036221512717e-04;
       Double_t current = (*dcsHandle)[0].magnetCurrent();
       bfield = current*currentToBFieldScaleFactor;
     } else {
       edm::ESHandle<MagneticField> magneticField;
       iSetup.get<IdealMagneticFieldRecord>().get(magneticField);
       const  MagneticField *mField = magneticField.product();
       bfield = mField->inTesla(GlobalPoint(0.,0.,0.)).z();
     }
     edm::Handle<reco::TrackCollection> ctfTracks;
     if ( iEvent.getByLabel("generalTracks", ctfTracks) ) {
       ConversionFinder cf;
       Bool_t isConv = cf.isElFromConversion(maxETelec, ctfTracks, bfield, dist_, dcot_);
       Bool_t isConv2 = cf.isElFromConversion(maxETelec2, ctfTracks, bfield, dist2_, dcot2_);
       if (isConv) maxETelec.addUserInt("PassConversionRejection",0);
       else maxETelec.addUserInt("PassConversionRejection",1);
       if (isConv2) maxETelec2.addUserInt("PassConversionRejection",0);
       else maxETelec2.addUserInt("PassConversionRejection",1);
     }
     else {
       std::cout << "WARNING! Track Collection with input name: generalTracks"
                 << " was not found. Conversion Rejection is not going to be"
                 << " applied!!!" << std::endl;
     }
   }
   //
   // get the met now:
   const pat::METCollection *pMet = patMET.product();
   const pat::METCollection::const_iterator met = pMet->begin();
   const pat::MET myMET = *met;
   Double_t metEt = met->et();
   //
   if (metEt < METCut_) {
     delete [] sorted;  delete [] et;
     return false; // RETURN: event MET < met cut
   }
   //
   //
   // if you have indeed reached this point then you have a zeeCandidate!!!
   pat::CompositeCandidate zeeCandidate;
   zeeCandidate.addDaughter(maxETelec, "electron1");
   zeeCandidate.addDaughter(maxETelec2, "electron2");
   zeeCandidate.addDaughter(myMET, "met");
   auto_ptr<pat::CompositeCandidateCollection> 
     selectedZeeCandidates(new pat::CompositeCandidateCollection);
   //
   //
   selectedZeeCandidates->push_back(zeeCandidate);
   //
   iEvent.put( selectedZeeCandidates, "selectedZeeCandidates");
   //
   // release your memory
   delete [] sorted;  delete [] et;
   //
   //cout << "This event has passed!" << endl;
   return true;
}

// ------------ method called once each job just after ending the event loop  -
void 
ZeeCandidateFilter::endJob() {
}

Bool_t ZeeCandidateFilter::isInFiducial(Double_t eta)
{
  if (TMath::Abs(eta) < BarrelMaxEta_) return true;
  else if (TMath::Abs(eta) < EndCapMaxEta_ && TMath::Abs(eta) > EndCapMinEta_)
    return true;
  return false;

}

//define this as a plug-in
DEFINE_FWK_MODULE(ZeeCandidateFilter);

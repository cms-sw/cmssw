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
// for conversion finder
#include "RecoEgamma/EgammaTools/interface/ConversionFinder.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
//
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
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
//     Bool_t passEleIDCuts(pat::Electron *ele);
    // ----------member data ---------------------------
    Double_t ETCut_;
    Double_t METCut_;

    edm::InputTag triggerCollectionTag_;
    edm::InputTag triggerEventTag_;
    std::string   hltpath_;
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
    edm::InputTag PrimaryVerticesCollection_;
    Double_t BarrelMaxEta_;
    Double_t EndCapMaxEta_;
    Double_t EndCapMinEta_;
    Bool_t useTriggerInfo_;
    Bool_t electronMatched2HLT_;
    Double_t electronMatched2HLT_DR_;
    
    Bool_t useValidFirstPXBHit_;
    Bool_t calculateValidFirstPXBHit_;
    Bool_t useConversionRejection_;
    Bool_t calculateConversionRejection_;
    Double_t dist_, dcot_;
    Bool_t useExpectedMissingHits_;
    Bool_t calculateExpectedMissingHits_;
    Int_t  maxNumberOfExpectedMissingHits_;
     
    Bool_t useValidFirstPXBHit2_;
    Bool_t calculateValidFirstPXBHit2_;
    Bool_t useConversionRejection2_;
    Bool_t calculateConversionRejection2_;
    Double_t dist2_, dcot2_;
    Bool_t useExpectedMissingHits2_;
    Bool_t calculateExpectedMissingHits2_;
    Int_t  maxNumberOfExpectedMissingHits2_;
    
    Bool_t dataMagneticFieldSetUp_;
    edm::InputTag dcsTag_;
    Bool_t useEcalDrivenElectrons_;
    Bool_t useSpikeRejection_;
    Double_t spikeCleaningSwissCrossCut_;
    Bool_t useHLTObjectETCut_;
    Double_t hltObjectETCut_;
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
    ETCut_ = iConfig.getUntrackedParameter<double>("ETCut");
    METCut_ = iConfig.getUntrackedParameter<double>("METCut");

    useEcalDrivenElectrons_ = iConfig.getUntrackedParameter<Bool_t>("useEcalDrivenElectrons", false);
    //
    // preselection criteria: hit pattern
    useValidFirstPXBHit_ =  iConfig.getUntrackedParameter<Bool_t>("useValidFirstPXBHit",false);
    calculateValidFirstPXBHit_ =  iConfig.getUntrackedParameter<Bool_t>("calculateValidFirstPXBHit",false);
    useConversionRejection_ =   iConfig.getUntrackedParameter<Bool_t>("useConversionRejection",false);
    calculateConversionRejection_ = iConfig.getUntrackedParameter<Bool_t>("calculateConversionRejection",false);
    dist_ = iConfig.getUntrackedParameter<Double_t>("conversionRejectionDist", 0.02);
    dcot_ = iConfig.getUntrackedParameter<Double_t>("conversionRejectionDcot", 0.02);
    dist2_ = iConfig.getUntrackedParameter<Double_t>("conversionRejectionDist2", 0.02);
    dcot2_ = iConfig.getUntrackedParameter<Double_t>("conversionRejectionDcot2", 0.02);
    dataMagneticFieldSetUp_ = iConfig.getUntrackedParameter<Bool_t>("dataMagneticFieldSetUp",false);
    if (dataMagneticFieldSetUp_) {
        dcsTag_ = iConfig.getUntrackedParameter<edm::InputTag>("dcsTag");
    }
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
      std::cout << "ZeeCandidateFilter: ERROR IN Configuration: vHltpathExtra and vHltpathFilterExtra"
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
    PrimaryVerticesCollection_ = iConfig.getUntrackedParameter<edm::InputTag>("PrimaryVerticesCollection");
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
    std::cout << "ZeeCandidateFilter: Running Zee Filter..." << std::endl;
    if (useTriggerInfo_) {
        std::cout << "ZeeCandidateFilter: HLT Path   " << hltpath_ << std::endl;
        std::cout << "ZeeCandidateFilter: HLT Filter " <<hltpathFilter_<<std::endl;
        if (useExtraTrigger_) {
            for (int itrig=0; itrig < (int) vHltpathExtra_.size(); ++itrig) {
	        std::cout << "ZeeCandidateFilter: OR " << vHltpathExtra_[itrig] 
		        << " with filter: " << vHltpathFilterExtra_[itrig] << std::endl;
            }
        }
    }
    else {
        std::cout << "ZeeCandidateFilter: Trigger info will not be used here"
                  << std::endl;
    }
    std::cout << "ZeeCandidateFilter: ET  > " << ETCut_ << std::endl;
    std::cout << "ZeeCandidateFilter: MET > " << METCut_ << std::endl;


    if (useEcalDrivenElectrons_) {
        std::cout << "ZeeCandidateFilter: Electron Candidate is required to"
                  << " be ecal driven" << std::endl;
    }
    if (electronMatched2HLT_&& useTriggerInfo_) {
        std::cout << "ZeeCandidateFilter: at least one electron is required to "
                  << "match an HLT object with DR < " << electronMatched2HLT_DR_
                  << std::endl;
    } else {
        std::cout << "ZeeCandidateFilter: Electron Candidates NOT required to "
                  << "match HLT object " << std::endl;
    }
    if (useValidFirstPXBHit_) {
        std::cout << "ZeeCandidateFilter: Electron Candidate required to have "
                  << "a valid hit in 1st PXB layer " << std::endl;
    }
    if (calculateValidFirstPXBHit_) {
        std::cout << "ZeeCandidateFilter: Info about whether there is a valid 1st layer PXB hit "
                  << "will be stored: you can access that later by "
                  << "myElec.userInt(\"PassValidFirstPXBHit\")==1" << std::endl;
    }
    if (useExpectedMissingHits_) {
        std::cout << "ZeeCandidateFilter: Electron Candidate required to have "
                  << " less than " << maxNumberOfExpectedMissingHits_
                  << " expected hits missing " << std::endl;
        ;
    }
    if (calculateExpectedMissingHits_) {
        std::cout << "ZeeCandidateFilter: Missing Hits from expected inner layers "
                  << "will be calculated and stored: you can access them later by "
                  << "myElec.userInt(\"NumberOfExpectedMissingHits\")"   << std::endl;
    }
    if (useConversionRejection_) {
        std::cout << "ZeeCandidateFilter: Electron Candidate required to pass "
                  << "EGAMMA Conversion Rejection criteria " << std::endl;
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
    if (useSpikeRejection_) {
        std::cout << "ZeeCandidateFilter: Spike Cleaning will be done with the Swiss Cross Criterion"
                  << " cutting at " << spikeCleaningSwissCrossCut_ << std::endl;
    }
    std::cout << "ZeeCandidateFilter: Fiducial Cut: " << std::endl;
    std::cout << "ZeeCandidateFilter:    BarrelMax: "<<BarrelMaxEta_<<std::endl;
    std::cout << "ZeeCandidateFilter:    EndcapMin: " << EndCapMinEta_
              << "  EndcapMax: " << EndCapMaxEta_
              <<std::endl;
    //
    // extra info in the event
    produces< pat::CompositeCandidateCollection >
    ("selectedZeeCandidates").setBranchAlias("selectedZeeCandidates");

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
    //
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
     UInt_t trigger_position_extra;
     if (trigger_position < trigger_size)
       passTrigger = (int) HLTResults->accept(trigger_position);
     if (useExtraTrigger_ && passTrigger==0) {
       for (int itrig=0; itrig < (int) vHltpathExtra_.size(); ++itrig) {
	 trigger_position_extra = triggerNames.triggerIndex(vHltpathExtra_[itrig]);
	 if (trigger_position_extra > 0) {
	   passTrigger = 1;
	   break;
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
       filterIndExtra.push_back( pHLT->filterIndex(vHltpathFilterExtra_[itrig]) );
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
    iEvent.getByLabel(metCollectionTag_, patMET);
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
    if (Nelecs <= 1) {
        //std::cout << "No more than 1 electrons found in this event" << std::endl;
        return false; // RETURN if less than 2 elecs in the event
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
    if (event_elec_number <=1) {
        //std::cout << "No more than 1 electrons in fiducial were found" << std::endl;
        return false; // RETURN if no more than 1 electron in fiducial
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
    // if the 2 highest electrons in the event has ET < ETCut_ return
    Int_t max_et_index = sorted[0];
    Int_t max_et_index2 = sorted[1];
    if (ETs[max_et_index]<ETCut_ || ETs[max_et_index2]<ETCut_) {
        delete [] sorted;
        delete [] et;
        return false; // RETURN: demand the highest ET electrons ET>ETcut
    }
    //
    // my electrons now:
    pat::Electron maxETelec = myElectrons[max_et_index];
    pat::Electron maxETelec2 = myElectrons[max_et_index2];
    //
    // demand that they are in fiducial:
    if (not isInFiducial(maxETelec.caloPosition().eta())) {
        delete [] sorted;
        delete [] et;
        return false; // RETURN highest ET electron is not in fiducial
    }
    if (not isInFiducial(maxETelec2.caloPosition().eta())) {
        delete [] sorted;
        delete [] et;
        return false; // RETURN 2nd highest ET electron is not in fiducial
    }
    // demand that they are ecal driven
    if (useEcalDrivenElectrons_) {
        if (not maxETelec.ecalDrivenSeed()) {
            delete [] sorted;
            delete [] et;
            return false; // RETURN highest ET electron is not ecal driven
        }
    }
//     if (useEcalDrivenElectrons2_) {
//         if (not maxETelec2.ecalDrivenSeed()) {
//             delete [] sorted;
//             delete [] et;
//             return false; // RETURN 2nd highest ET electron is not ecal driven
//         }
//     }
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
            delete [] sorted;
            delete [] et;
            return false; // RETURN highest ET electron is a spike
        }
    }
    if (useSpikeRejection_ && maxETelec2.isEB()) {
        edm::Handle<EcalRecHitCollection> recHits;
        if (maxETelec2.isEB())
            iEvent.getByLabel(ebRecHits_, recHits);
        else
            iEvent.getByLabel(eeRecHits_, recHits);
        const EcalRecHitCollection *myRecHits = recHits.product();
        const DetId seedId = maxETelec2.superCluster()->seed()->seed();
        EcalSeverityLevelAlgo severity;
        Double_t swissCross = severity.swissCross(seedId, *myRecHits);
        if (swissCross > spikeCleaningSwissCrossCut_) {
            delete [] sorted;
            delete [] et;
            return false; // RETURN 2nd highest ET electron is a spike
        }
    }
    // add the primary vtx information in the electron:
    edm::Handle< std::vector<reco::Vertex> > pVtx;
    iEvent.getByLabel(PrimaryVerticesCollection_, pVtx);
    const std::vector<reco::Vertex> Vtx = *(pVtx.product());
    double pv_x1 = -999999.;
    double pv_y1 = -999999.;
    double pv_z1 = -999999.;
    double ele_tip_pv1 = -999999.;
    if (Vtx.size() >=1) {
        pv_x1 = Vtx[0].position().x();
        pv_y1 = Vtx[0].position().y();
        pv_z1 = Vtx[0].position().z();
        ele_tip_pv1 = -maxETelec.gsfTrack()->dxy(Vtx[0].position());
    }
    //
    maxETelec.addUserFloat("pv_x", Float_t(pv_x1));
    maxETelec.addUserFloat("pv_x", Float_t(pv_y1));
    maxETelec.addUserFloat("pv_z", Float_t(pv_z1));
    maxETelec.addUserFloat("ele_tip_pv", Float_t(ele_tip_pv1));

    edm::Handle< std::vector<reco::Vertex> > pVtx2;
    iEvent.getByLabel(PrimaryVerticesCollection_, pVtx2);
    const std::vector<reco::Vertex> Vtx2 = *(pVtx2.product());
    double pv_x2 = -999999.;
    double pv_y2 = -999999.;
    double pv_z2 = -999999.;
    double ele_tip_pv2 = -999999.;
    if (Vtx2.size() >=1) {
        pv_x2 = Vtx2[0].position().x();
        pv_y2 = Vtx2[0].position().y();
        pv_z2 = Vtx2[0].position().z();
        ele_tip_pv2 = -maxETelec2.gsfTrack()->dxy(Vtx2[0].position());
    }
    //
    maxETelec2.addUserFloat("pv_x", Float_t(pv_x1));
    maxETelec2.addUserFloat("pv_x", Float_t(pv_y1));
    maxETelec2.addUserFloat("pv_z", Float_t(pv_z1));
    maxETelec2.addUserFloat("ele_tip_pv", Float_t(ele_tip_pv2));
    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    // special pre-selection requirements ^^^
    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    // hit pattern and conversion rejection

    if (useValidFirstPXBHit_ || calculateValidFirstPXBHit_) {
        Bool_t fail =
            not maxETelec.gsfTrack()->hitPattern().hasValidHitInFirstPixelBarrel();
        if (useValidFirstPXBHit_ && fail)
        {
            delete [] sorted;
            delete [] et;
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

    if (useValidFirstPXBHit2_ || calculateValidFirstPXBHit2_) {
        Bool_t fail =
            not maxETelec2.gsfTrack()->hitPattern().hasValidHitInFirstPixelBarrel();
        if (useValidFirstPXBHit2_ && fail)
        {
            delete [] sorted;
            delete [] et;
            //std::cout << "Filter: there is no valid hit in 1st layer PXB" << std::endl;
            return false;
        }
        if (calculateValidFirstPXBHit2_) {
            std::string vfpx("PassValidFirstPXBHit");
            if (fail)
                maxETelec2.addUserInt(vfpx,0);
            else
                maxETelec2.addUserInt(vfpx,1);
        }
    }

    if (useExpectedMissingHits_ || calculateExpectedMissingHits_) {
        Int_t numberOfInnerHits = (Int_t) maxETelec.gsfTrack()->trackerExpectedHitsInner().numberOfHits();
        if (numberOfInnerHits > maxNumberOfExpectedMissingHits_
                && useExpectedMissingHits_) {
            delete [] sorted;
            delete [] et;
            return false;
        }
        if (calculateExpectedMissingHits_) {
            maxETelec.addUserInt("NumberOfExpectedMissingHits",numberOfInnerHits);
        }
    }

    if (useExpectedMissingHits2_ || calculateExpectedMissingHits2_) {
        Int_t numberOfInnerHits = (Int_t) maxETelec2.gsfTrack()->trackerExpectedHitsInner().numberOfHits();
        if (numberOfInnerHits > maxNumberOfExpectedMissingHits2_
                && useExpectedMissingHits2_) {
            delete [] sorted;
            delete [] et;
            return false;
        }
        if (calculateExpectedMissingHits2_) {
            maxETelec2.addUserInt("NumberOfExpectedMissingHits",numberOfInnerHits);
        }
    }

    if (useConversionRejection_ || calculateConversionRejection_) {
        // use of conversion rejection as it is implemented in egamma
        // you have to get the general track collection to do that
        // WARNING! you have to supply the correct B-field in Tesla
        // the magnetic field
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
            ConversionFinder convFinder;
            ConversionInfo convInfo =
                convFinder.getConversionInfo(maxETelec, ctfTracks, bfield);
            Double_t dist = convInfo.dist();
            Double_t dcot = convInfo.dcot();
            Bool_t isConv = ( fabs(dist) < dist_ ) && ( fabs(dcot) < dcot_ );
            //std::cout << "Filter: for this elec the conversion says " << isConv << std::endl;
            if (isConv && useConversionRejection_) {
                delete [] sorted;
                delete [] et;
                return false;
            }
            if (calculateConversionRejection_) {
                maxETelec.addUserFloat("Dist", Float_t(dist));
                maxETelec.addUserFloat("Dcot", Float_t(dcot));
                if (isConv)
                    maxETelec.addUserInt("PassConversionRejection",0);
                else
                    maxETelec.addUserInt("PassConversionRejection",1);
            }
        } else {
            std::cout << "WARNING! Track Collection with input name: generalTracks"
                      << " was not found. Conversion Rejection is not going to be"
                      << " applied!!!" << std::endl;
        }

    }

    if (useConversionRejection2_ || calculateConversionRejection2_) {
        // use of conversion rejection as it is implemented in egamma
        // you have to get the general track collection to do that
        // WARNING! you have to supply the correct B-field in Tesla
        // the magnetic field
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
            ConversionFinder convFinder;
            ConversionInfo convInfo =
                convFinder.getConversionInfo(maxETelec2, ctfTracks, bfield);
            Double_t dist = convInfo.dist();
            Double_t dcot = convInfo.dcot();
            Bool_t isConv = ( fabs(dist) < dist2_ ) && ( fabs(dcot) < dcot2_ );
            //std::cout << "Filter: for this elec the conversion says " << isConv << std::endl;
            if (isConv && useConversionRejection2_) {
                delete [] sorted;
                delete [] et;
                return false;
            }
            if (calculateConversionRejection2_) {
                maxETelec2.addUserFloat("Dist", Float_t(dist));
                maxETelec2.addUserFloat("Dcot", Float_t(dcot));
                if (isConv)
                    maxETelec2.addUserInt("PassConversionRejection",0);
                else
                    maxETelec2.addUserInt("PassConversionRejection",1);
            }
        } else {
            std::cout << "WARNING! Track Collection with input name: generalTracks"
                      << " was not found. Conversion Rejection is not going to be"
                      << " applied!!!" << std::endl;
        }

    }

   //
   //std::cout << "HLT matching starts" << std::endl;
   if (electronMatched2HLT_ && useTriggerInfo_) {
     Double_t matched_dr_distance = 999999.;
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
	       reco::deltaR(maxETelec.eta(),maxETelec.phi(),TO.eta(),TO.phi());
	     //std::cout << "-->found dr=" << dr_ele_HLT << std::endl;
	     if (TMath::Abs(dr_ele_HLT) < matched_dr_distance) {
	       matched_dr_distance = dr_ele_HLT;
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
     std::string  HLTPath[18];
     HLTPath[0 ] = "HLT_Photon10_L1R"      ;
     HLTPath[1 ] = "HLT_Photon15_L1R"      ;
     HLTPath[2 ] = "HLT_Photon20_L1R"      ;
     HLTPath[3 ] = "HLT_Photon15_TrackIso_L1R";
     HLTPath[4 ] = "HLT_Photon15_LooseEcalIso_L1R";
     HLTPath[5 ] = "HLT_Photon30_L1R_8E29" ;
     HLTPath[6 ] = "HLT_Photon30_L1R_8E29" ;
     HLTPath[7 ] = "HLT_Ele10_LW_L1R"      ;
     HLTPath[8 ] = "HLT_Ele15_LW_L1R"      ;
     HLTPath[9 ] = "HLT_Ele20_LW_L1R"      ;
     HLTPath[10] = "HLT_Ele10_LW_EleId_L1R";
     HLTPath[11] = "HLT_Ele15_SiStrip_L1R" ;
     HLTPath[12] = "HLT_IsoTrackHB_8E29"   ;
     HLTPath[13] = "HLT_IsoTrackHE_8E29"   ;
     HLTPath[14] = "HLT_DiJetAve15U_8E29"  ;
     HLTPath[15] = "HLT_MET45"             ;
     HLTPath[16] = "HLT_L1MET20"           ;
     HLTPath[17] = "HLT_MET100"            ;
     //	      
     edm::InputTag HLTFilterType[15];
     HLTFilterType[0 ]= edm::InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt10HcalIsolFilter","",process);    // HLT_Photon10_L1R
     HLTFilterType[1 ]= edm::InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter" ,"",process);   // HLT_Photon15_L1R
     HLTFilterType[2 ]= edm::InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt20HcalIsolFilter" ,"",process);   // HLT_Photon20_L1R
     HLTFilterType[3 ]= edm::InputTag("hltL1NonIsoSinglePhotonEt15HTITrackIsolFilter","",process);         // HLT_Photon15_TrackIso_L1R
     HLTFilterType[4 ]= edm::InputTag("hltL1NonIsoSinglePhotonEt15LEIHcalIsolFilter","",process);          // HLT_Photon15_LooseEcalIso_L1R
     HLTFilterType[5 ]= edm::InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt15EtFilterESet308E29","",process);// HLT_Photon30_L1R_8E29
     HLTFilterType[6 ]= edm::InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter","",process);    // HLT_Photon30_L1R_8E29
     HLTFilterType[7 ]= edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt10PixelMatchFilter","",process); 
     HLTFilterType[8 ]= edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt15PixelMatchFilter","",process);
     HLTFilterType[9 ]= edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt15EtFilterESet20","",process);
     HLTFilterType[10]= edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdDphiFilter","",process);
     HLTFilterType[11]= edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronSiStripEt15PixelMatchFilter","",process);
     HLTFilterType[12]= edm::InputTag("hltIsolPixelTrackL3FilterHB8E29","",process);
     HLTFilterType[13]= edm::InputTag("hltIsolPixelTrackL2FilterHE8E29","",process);
     HLTFilterType[14]= edm::InputTag("hltL1sDiJetAve15U8E29","",process);
     //		      
     Int_t triggerDecision =0;
     UInt_t trigger_size = HLTResults->size();
     for (Int_t i=0; i<18; ++i) {
       const edm::TriggerNames & triggerNames = iEvent.triggerNames(*HLTResults);
       UInt_t trigger_position = triggerNames.triggerIndex(HLTPath[i]);
       Int_t  passTrigger = 0;
       if (trigger_position < trigger_size) {
	 passTrigger = (Int_t) HLTResults->accept(trigger_position);
       }
       if (passTrigger >0) {
	 if (i>=15) {triggerDecision += Int_t(TMath::Power(2,i));}
	 else {
	   const Int_t myfilterInd = pHLT->filterIndex(HLTFilterType[i]);
	   if (myfilterInd != nF) {
	     triggerDecision +=  Int_t(TMath::Power(2,i));
	   }
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
        delete [] sorted;
        delete [] et;
        //std::cout << "MET is " << metEt << std::endl;
        return false;  // RETURN if MET is < Metcut
    }
    //
    //
    // if you have indeed reached this point then you have a zeeCandidate!!!
    pat::CompositeCandidate zeeCandidate;
    zeeCandidate.addDaughter(maxETelec, "electron1");
    zeeCandidate.addDaughter(maxETelec2, "electron2");  
    zeeCandidate.addDaughter(theMET, "met");
    zeeCandidate.addDaughter(thePfMET, "pfmet");
    zeeCandidate.addDaughter(theTcMET, "tcmet");
    auto_ptr<pat::CompositeCandidateCollection>
    selectedZeeCandidates(new pat::CompositeCandidateCollection);
    selectedZeeCandidates->push_back(zeeCandidate);
    //
    iEvent.put( selectedZeeCandidates, "selectedZeeCandidates");

    //
    // release your memory
    delete [] sorted;
    delete [] et;
    //
    //std::cout << "Filter accepts this event" << std::endl;
    //
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

// Bool_t ZeeCandidateFilter::passEleIDCuts(pat::Electron *ele)
// {
//     if (not useVetoSecondElectronID_)  return true;
//     if (not ele->isElectronIDAvailable(vetoSecondElectronIDType_)) {
//         std::cout << "ZeeCandidateFilter: request ignored: 2nd electron ID type "
//                   << "not found in electron object" << std::endl;
//         return true;
//     }
//     if (vetoSecondElectronIDSign_ == ">") {
//         if (ele->electronID(vetoSecondElectronIDType_)>vetoSecondElectronIDValue_)
//             return true;
//         else return false;
//     }
//     else if (vetoSecondElectronIDSign_ == "<") {
//         if (ele->electronID(vetoSecondElectronIDType_)<vetoSecondElectronIDValue_)
//             return true;
//         else return false;
//     }
//     else {
//         if (TMath::Abs(ele->electronID(vetoSecondElectronIDType_)-
//                        vetoSecondElectronIDValue_) < 0.1)
//             return true;
//         else return false;
//     }
// }

//define this as a plug-in
DEFINE_FWK_MODULE(ZeeCandidateFilter);

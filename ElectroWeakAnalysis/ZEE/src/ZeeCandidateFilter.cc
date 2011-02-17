// -*- C++ -*-
//
// Package:    ZeeCandidateFilter
// Class:      ZeeCandidateFilter
//
/**\class ZeeCandidateFilter ZeeCandidateFilter.cc ElectroWeakAnalysis/ZEE/src/ZeeCandidateFilter.cc

 Author(s):     Stilianos Kesisoglou - Institute of Nuclear Physics NCSR Demokritos (current author) 
                Nikolaos Rompotis    - Imperial College London                      (original author) 
                
 Contact:       Stilianos.Kesisoglou@cern.ch

 Description: <one line class summary>

 Implementation:

    This class contains a filter that searches the event and finds whether it fulfills the Z Candidate Criteria.
    If it fullfills them it creates a ZeeCandidate and stores it in the event.

    Definition of the Zee Caldidate:
    * event that passes the trigger
    * has 2 GSF electrons in fiducial with ET greater than a (configurable) threshold
    * at least one of them matched to an HLT Object (configurable) with DR < (configurable)

 Changes Log:

 12 Feb 2009    First Release of the code for CMSSW_2_2_X
 17 Sep 2009    First Release for CMSSW_3_1_X
 09 Dec 2009    Option to ignore trigger
 
 25 Feb 2010    Added options to use Conversion Rejection, Expected missing hits and valid hit at first PXB
 
                Added option to calculate these criteria and store them in the pat electron object this is done by setting in the configuration the flags
                
                    calculateValidFirstPXBHit = true
                    calculateConversionRejection = true
                    calculateExpectedMissinghits = true
                    
                Then the code calculates them and you can access all these from pat::Electron
                
                    myElec.userInt("PassValidFirstPXBHit")              0 fail, 1 passes
                    myElec.userInt("PassConversionRejection")           0 fail, 1 passes
                    myElec.userInt("NumberOfExpectedMissingHits")       the number of lost hits
                    
 28 May 2010    Implementation of Spring10 selections

 25 Jun 2010    Author change (Nikolaos Rompotis -> Stilianos Kesisoglou)
                Preparation of the code for the ICHEP 2010 presentation of the ElectroWeak results.
                
 04 Nov 2010    Changes to all variable types from C/C++ to ROOT ones (int -> Int_t etc..)
                Code modification to apply common or separate electron criteria.
                Addition of various print-out statements to follow code processing.

 08 Feb 2011    Modifications to allow for a "vector-like" treatment of all triggers

 13 Feb 2011    Modifications to transfer the vertex code from here to "Ploter"
                Modification to retrieve the dcot/dist variables from the EGamma code.
                
*/

#ifndef ZeeCandidateFilter_H
#define ZeeCandidateFilter_H

// System include files
#include <memory>

// User include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>
#include <iostream>
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "TString.h"
#include "TMath.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "DataFormats/PatCandidates/interface/TriggerObject.h"

// For conversion finder
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


//  Class Declaration
//  -----------------
//
class ZeeCandidateFilter : public edm::EDFilter {

public:

    explicit ZeeCandidateFilter(const edm::ParameterSet&);

    ~ZeeCandidateFilter();

private:

    virtual Bool_t filter(edm::Event&, const edm::EventSetup&);

    virtual void endJob() ;

    Bool_t isInFiducial(Double_t eta);

    //  -----   Data Members    -----

    Double_t                    ETCut_                                      ;
    Double_t                    METCut_                                     ;

    Bool_t                      useEcalDrivenElectrons_                     ;

    Bool_t                      useSameSelectionOnBothElectrons_            ;

    /*  Electron 1  */
    Bool_t                      useValidFirstPXBHit1_                       ;
    Bool_t                      calculateValidFirstPXBHit1_                 ;
    Bool_t                      useConversionRejection1_                    ;
    Bool_t                      useExpectedMissingHits1_                    ;
    Bool_t                      calculateExpectedMissingHits1_              ;
    Int_t                       maxNumberOfExpectedMissingHits1_            ;

    /*  Electron 2  */
    Bool_t                      useValidFirstPXBHit2_                       ;
    Bool_t                      calculateValidFirstPXBHit2_                 ;
    Bool_t                      useConversionRejection2_                    ;
    Bool_t                      useExpectedMissingHits2_                    ;
    Bool_t                      calculateExpectedMissingHits2_              ;
    Int_t                       maxNumberOfExpectedMissingHits2_            ;

    /*  Electron 1  */
    Double_t                    dist1_                                      ;
    Double_t                    dcot1_                                      ;

    /*  Electron 2  */
    Double_t                    dist2_                                      ;
    Double_t                    dcot2_                                      ;

    Double_t                    BarrelMaxEta_                               ;
    Double_t                    EndCapMaxEta_                               ;
    Double_t                    EndCapMinEta_                               ;

    edm::InputTag               triggerCollectionTag_                       ;
    edm::InputTag               triggerEventTag_                            ;

    std::vector<std::string>    vHltPath_                                   ;
    std::vector<edm::InputTag>  vHltPathFilter_                             ;

    std::vector<Int_t>          vUseHltObjectETCut_                         ;
    std::vector<Double_t>       vHltObjectETCut_                            ;

    Bool_t                      useTriggerInfo_                             ;
    Bool_t                      electronMatched2HLT_                        ;
    Double_t                    electronMatched2HLT_DR_                     ;

    edm::InputTag               electronCollectionTag_                      ;

    edm::InputTag               metCollectionTag_                           ;
    edm::InputTag               pfMetCollectionTag_                         ;
    edm::InputTag               tcMetCollectionTag_                         ;

    edm::InputTag               ebRecHits_                                  ;
    edm::InputTag               eeRecHits_                                  ;

    Bool_t                      useSpikeRejection_                          ;

    Double_t                    spikeCleaningSwissCrossCut_                 ;

};

#endif

//  Constants, Enums and Typedefs
//  -----------------------------
//

//  Static Data Member Definitions
//  ------------------------------
//

//  Constructors and Destructor
//  ---------------------------
//
ZeeCandidateFilter::ZeeCandidateFilter(const edm::ParameterSet& iConfig)
{
    //-------------------------------------//
    //         INITIALIZATION              //
    //-------------------------------------//

    //  Cuts
    //  ----
    ETCut_  = iConfig.getUntrackedParameter<Double_t>("ETCut");
    METCut_ = iConfig.getUntrackedParameter<Double_t>("METCut");

    useEcalDrivenElectrons_ = iConfig.getUntrackedParameter<Bool_t>("useEcalDrivenElectrons", true);

    useSameSelectionOnBothElectrons_ = iConfig.getUntrackedParameter<Bool_t>("useSameSelectionOnBothElectrons",true);
    //--------------------------------------------------------------------------------------------------------------------


    if ( useSameSelectionOnBothElectrons_ ) {
    
        //  Preselection Criteria: Hit Pattern
        //  ----------------------------------
        //
        /*  Electron 1  */
        useValidFirstPXBHit1_             =  iConfig.getUntrackedParameter<Bool_t>("useValidFirstPXBHit0",false);
        calculateValidFirstPXBHit1_       =  iConfig.getUntrackedParameter<Bool_t>("calculateValidFirstPXBHit0",false);
        useConversionRejection1_          =  iConfig.getUntrackedParameter<Bool_t>("useConversionRejection0",true);
        useExpectedMissingHits1_          =  iConfig.getUntrackedParameter<Bool_t>("useExpectedMissingHits0",false);
        calculateExpectedMissingHits1_    =  iConfig.getUntrackedParameter<Bool_t>("calculateExpectedMissingHits0",false);
        maxNumberOfExpectedMissingHits1_  =  iConfig.getUntrackedParameter<Int_t>("maxNumberOfExpectedMissingHits0",0);

        /*  Electron 2  */
        useValidFirstPXBHit2_             =  iConfig.getUntrackedParameter<Bool_t>("useValidFirstPXBHit0",false);
        calculateValidFirstPXBHit2_       =  iConfig.getUntrackedParameter<Bool_t>("calculateValidFirstPXBHit0",false);
        useConversionRejection2_          =  iConfig.getUntrackedParameter<Bool_t>("useConversionRejection0",true);
        useExpectedMissingHits2_          =  iConfig.getUntrackedParameter<Bool_t>("useExpectedMissingHits0",false);
        calculateExpectedMissingHits2_    =  iConfig.getUntrackedParameter<Bool_t>("calculateExpectedMissingHits0",false);
        maxNumberOfExpectedMissingHits2_  =  iConfig.getUntrackedParameter<Int_t>("maxNumberOfExpectedMissingHits0",0);
        //--------------------------------------------------------------------------------------------------------------------


        //  Conversion Rejection Variables
        //  ------------------------------
        //
        /*  Electron 1 and 2 */
        Double_t dist0_D  = 0.02 ;
        Double_t dcot0_D  = 0.02 ;

        dist1_ = iConfig.getUntrackedParameter<Double_t>("conversionRejectionDist1", dist0_D);
        dcot1_ = iConfig.getUntrackedParameter<Double_t>("conversionRejectionDcot1", dcot0_D);

        dist2_ = iConfig.getUntrackedParameter<Double_t>("conversionRejectionDist2", dist0_D);
        dcot2_ = iConfig.getUntrackedParameter<Double_t>("conversionRejectionDcot2", dcot0_D);
        
    }
    else {
    
        //  Preselection Criteria: Hit Pattern
        //  ----------------------------------
        //
        /*  Electron 1  */
        useValidFirstPXBHit1_             =  iConfig.getUntrackedParameter<Bool_t>("useValidFirstPXBHit1",false);
        calculateValidFirstPXBHit1_       =  iConfig.getUntrackedParameter<Bool_t>("calculateValidFirstPXBHit1",false);
        useConversionRejection1_          =  iConfig.getUntrackedParameter<Bool_t>("useConversionRejection1",true);
        useExpectedMissingHits1_          =  iConfig.getUntrackedParameter<Bool_t>("useExpectedMissingHits1",false);
        calculateExpectedMissingHits1_    =  iConfig.getUntrackedParameter<Bool_t>("calculateExpectedMissingHits1",false);
        maxNumberOfExpectedMissingHits1_  =  iConfig.getUntrackedParameter<Int_t>("maxNumberOfExpectedMissingHits1",0);

        /*  Electron 2  */
        useValidFirstPXBHit2_             =  iConfig.getUntrackedParameter<Bool_t>("useValidFirstPXBHit2",false);
        calculateValidFirstPXBHit2_       =  iConfig.getUntrackedParameter<Bool_t>("calculateValidFirstPXBHit2",false);
        useConversionRejection2_          =  iConfig.getUntrackedParameter<Bool_t>("useConversionRejection2",true);
        useExpectedMissingHits2_          =  iConfig.getUntrackedParameter<Bool_t>("useExpectedMissingHits2",false);
        calculateExpectedMissingHits2_    =  iConfig.getUntrackedParameter<Bool_t>("calculateExpectedMissingHits2",false);
        maxNumberOfExpectedMissingHits2_  =  iConfig.getUntrackedParameter<Int_t>("maxNumberOfExpectedMissingHits2",0);
        //--------------------------------------------------------------------------------------------------------------------


        //  Conversion Rejection Variables
        //  ------------------------------
        //
        /*  Electron 1  */
        Double_t dist1_D  = 0.02 ;
        Double_t dcot1_D  = 0.02 ;

        dist1_ = iConfig.getUntrackedParameter<Double_t>("conversionRejectionDist1", dist1_D);
        dcot1_ = iConfig.getUntrackedParameter<Double_t>("conversionRejectionDcot1", dcot1_D);

        /*  Electron 2  */
        Double_t dist2_D  = 0.02 ;
        Double_t dcot2_D  = 0.02 ;

        dist2_ = iConfig.getUntrackedParameter<Double_t>("conversionRejectionDist2", dist2_D);
        dcot2_ = iConfig.getUntrackedParameter<Double_t>("conversionRejectionDcot2", dcot2_D);
        
    }
    //--------------------------------------------------------------------------------------------------------------------


    //  Detector Fiducial Cuts
    //  ----------------------
    //
    Double_t BarrelMaxEta_D = 1.4442 ;
    Double_t EndCapMinEta_D = 1.5660 ;
    Double_t EndCapMaxEta_D = 2.5000 ;

    BarrelMaxEta_ = iConfig.getUntrackedParameter<Double_t>("BarrelMaxEta", BarrelMaxEta_D);
    EndCapMaxEta_ = iConfig.getUntrackedParameter<Double_t>("EndCapMaxEta", EndCapMaxEta_D);
    EndCapMinEta_ = iConfig.getUntrackedParameter<Double_t>("EndCapMinEta", EndCapMinEta_D);
    //--------------------------------------------------------------------------------------------------------------------


    //  Trigger Related
    //  ---------------
    //
    triggerCollectionTag_ = iConfig.getUntrackedParameter<edm::InputTag>("triggerCollectionTag");
    triggerEventTag_      = iConfig.getUntrackedParameter<edm::InputTag>("triggerEventTag");

    vHltPath_       = iConfig.getUntrackedParameter< std::vector<std::string> >("vHltPath");
    vHltPathFilter_ = iConfig.getUntrackedParameter< std::vector<edm::InputTag> >("vHltPathFilter");

    vUseHltObjectETCut_   = iConfig.getUntrackedParameter< std::vector<Int_t> >("vUseHltObjectETCut");
    vHltObjectETCut_      = iConfig.getUntrackedParameter< std::vector<Double_t> >("vHltObjectETCut");

    //--------------------------------------------------------------------------------------------------------------------


    //  Trigger Matching Related
    //  ------------------------
    //
    useTriggerInfo_         = iConfig.getUntrackedParameter<Bool_t>("useTriggerInfo",true);
    electronMatched2HLT_    = iConfig.getUntrackedParameter<Bool_t>("electronMatched2HLT");
    electronMatched2HLT_DR_ = iConfig.getUntrackedParameter<Double_t>("electronMatched2HLT_DR");
    //--------------------------------------------------------------------------------------------------------------------


    //  Electrons, MET's Vtx's and other
    //  --------------------------------
    //
    electronCollectionTag_  = iConfig.getUntrackedParameter<edm::InputTag>("electronCollectionTag");

    metCollectionTag_   = iConfig.getUntrackedParameter<edm::InputTag>("metCollectionTag");
    pfMetCollectionTag_ = iConfig.getUntrackedParameter<edm::InputTag>("pfMetCollectionTag");
    tcMetCollectionTag_ = iConfig.getUntrackedParameter<edm::InputTag>("tcMetCollectionTag");

    ebRecHits_ = iConfig.getUntrackedParameter<edm::InputTag>("ebRecHits");
    eeRecHits_ = iConfig.getUntrackedParameter<edm::InputTag>("eeRecHits");
    //--------------------------------------------------------------------------------------------------------------------


    //  Spike Cleaning
    //  --------------
    //
    useSpikeRejection_ = iConfig.getUntrackedParameter<Bool_t>("useSpikeRejection");

    if ( useSpikeRejection_ ) {
        spikeCleaningSwissCrossCut_ = iConfig.getUntrackedParameter<Double_t>("spikeCleaningSwissCrossCut");
    }
    //--------------------------------------------------------------------------------------------------------------------


    //-------------------------------------//
    //         SUMMARY PRINTOUT            //
    //-------------------------------------//

    std::cout << "ZeeCandidateFilter: Running ZeeCandidateFilter..." << std::endl;

    if ( useTriggerInfo_ ) {
        for (Int_t itrig=0; itrig < (Int_t)vHltPath_.size(); ++itrig ) {        
            std::cout << "ZeeCandidateFilter: Trigger Information" << std::endl;            
            std::cout << "ZeeCandidateFilter:"
                      << " HLT Path "         << vHltPath_[itrig] 
                      << " with HLT Filter: " << vHltPathFilter_[itrig]
                      << " using EtCut: "     << vUseHltObjectETCut_[itrig]
                      << " EtCut Value: "     << vHltObjectETCut_[itrig]
                      << std::endl;
        }
    }
    else {
        std::cout << "ZeeCandidateFilter: Trigger info will not be used here" << std::endl;
    }

    std::cout << "ZeeCandidateFilter: ET  > " << ETCut_ << std::endl;
    std::cout << "ZeeCandidateFilter: MET > " << METCut_ << std::endl;


    if ( useEcalDrivenElectrons_ ) {
        std::cout << "ZeeCandidateFilter: Electron Candidate(s) is required to be ecal driven" << std::endl;
    }

    if ( electronMatched2HLT_ && useTriggerInfo_ ) {
        std::cout << "ZeeCandidateFilter: At least one electron is required to match an HLT object with DR < " << electronMatched2HLT_DR_ << std::endl;
    }
    else {
        std::cout << "ZeeCandidateFilter: Electron Candidates NOT required to match HLT object " << std::endl;
    }

    if ( useValidFirstPXBHit1_ ) {
        std::cout << "ZeeCandidateFilter: Electron Candidate #1 required to have a valid hit in 1st PXB layer " << std::endl;
    }

    if ( useValidFirstPXBHit2_ ) {
        std::cout << "ZeeCandidateFilter: Electron Candidate #2 required to have a valid hit in 1st PXB layer " << std::endl;
    }

    if ( calculateValidFirstPXBHit1_ ) {
        std::cout << "ZeeCandidateFilter: Info about whether there is a valid 1st layer PXB hit for electron candidate #1 will be stored: you can access that later by myElec.userInt(\"PassValidFirstPXBHit\")==1" << std::endl;
    }

    if ( calculateValidFirstPXBHit2_ ) {
        std::cout << "ZeeCandidateFilter: Info about whether there is a valid 1st layer PXB hit for electron candidate #2 will be stored: you can access that later by myElec.userInt(\"PassValidFirstPXBHit\")==1" << std::endl;
    }

    if ( useExpectedMissingHits1_ ) {
        std::cout << "ZeeCandidateFilter: Electron Candidate #1 is required to have less than " << maxNumberOfExpectedMissingHits1_ << " expected hits missing " << std::endl;
    }

    if ( useExpectedMissingHits2_ ) {
        std::cout << "ZeeCandidateFilter: Electron Candidate #2 is required to have less than " << maxNumberOfExpectedMissingHits2_ << " expected hits missing " << std::endl;
    }

    if ( calculateExpectedMissingHits1_ ) {
        std::cout << "ZeeCandidateFilter: Missing Hits from expected inner layers for electron candidate #1 will be calculated and stored: you can access them later by myElec.userInt(\"NumberOfExpectedMissingHits\")"   << std::endl;
    }

    if ( calculateExpectedMissingHits2_ ) {
        std::cout << "ZeeCandidateFilter: Missing Hits from expected inner layers for electron candidate #2 will be calculated and stored: you can access them later by myElec.userInt(\"NumberOfExpectedMissingHits\")"   << std::endl;
    }

    if ( useConversionRejection1_ ) {
        std::cout << "ZeeCandidateFilter: Electron Candidate #1 is required to pass EGAMMA Conversion Rejection criteria" << std::endl;
    }

    if ( useConversionRejection2_ ) {
        std::cout << "ZeeCandidateFilter: Electron Candidate #2 is required to pass EGAMMA Conversion Rejection criteria" << std::endl;
    }

    if ( useSpikeRejection_ ) {
        std::cout << "ZeeCandidateFilter: Spike Cleaning will be done with the Swiss Cross Criterion cutting at " << spikeCleaningSwissCrossCut_ << std::endl;
    }

    std::cout << "ZeeCandidateFilter: Fiducial Cut: "                                                      << std::endl;
    std::cout << "ZeeCandidateFilter:    BarrelMax: " << BarrelMaxEta_                                     << std::endl;
    std::cout << "ZeeCandidateFilter:    EndcapMin: " << EndCapMinEta_ << "  EndcapMax: " << EndCapMaxEta_ << std::endl;

    //
    //------------------------------------------//
    //         EXTRA INFO IN THE EVENT          //
    //------------------------------------------//
    //
    produces<pat::CompositeCandidateCollection>("selectedZeeCandidates").setBranchAlias("selectedZeeCandidates");

}


ZeeCandidateFilter::~ZeeCandidateFilter()
{
    // do anything here that needs to be done at desctruction time
    // (e.g. close files, deallocate resources etc.)
}


//  Member Functions
//  ----------------
//

// ------------ method called on each new Event  ------------
Bool_t ZeeCandidateFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    using namespace edm;
    using namespace std;
    using namespace pat;


    std::cout << "FILTER-MSG: Begin Processing ... "
              << "Run = "   << iEvent.run() << " " 
              << "Lumi = "  << (Int_t) iEvent.luminosityBlock() << " " 
              << "Event = " << iEvent.eventAuxiliary().event() << " " 
              << std::endl;


    /***    TRIGGER REQUIREMENT - Event should pass the trigger, otherwise no zee candidate     ***/

    edm::Handle<edm::TriggerResults> HLTResults;
    iEvent.getByLabel(triggerCollectionTag_, HLTResults);

    Int_t passTrigger = 0;

    if ( HLTResults.isValid() ) {

        const edm::TriggerNames & triggerNames = iEvent.triggerNames(*HLTResults);

        UInt_t trigger_size = HLTResults->size();
        
            for (Int_t itrig=0; itrig < (Int_t)vHltPath_.size(); ++itrig ) {
            
              UInt_t trigger_position = triggerNames.triggerIndex(vHltPath_[itrig]);
            
              if ( trigger_position < trigger_size ) {
                  passTrigger = (Int_t)HLTResults->accept(trigger_position);
              }
              
              //  Tested TriggerPath firing results printout
              std::cout << "SK_HLT_INFO"
                        << " | " << "trigger_size = "               << trigger_size 
                        << " | " << "vHltPath_.size() = "           << vHltPath_.size()
                        << " | " << "vHltPath_[" << itrig << "] = " << vHltPath_[itrig] 
                        << " | " << "trigger_position = "           << trigger_position 
                        << " | " << "passTrigger = "                << passTrigger
              << std::endl;

                if ( passTrigger > 0 ) { break ; }
                
            }   //  for Loop
        
    }
    else {  std::cout << "TriggerResults are missing from this event.." << std::endl;
        if ( useTriggerInfo_ ) {
            return false;    // RETURN if trigger is missing
        }
    }

    if ( passTrigger == 0 && useTriggerInfo_ ) {    std::cout << "No HLT Path is firing in this event" << std::endl;
        return false; // RETURN if event fails the trigger
    }
    
    edm::Handle<trigger::TriggerEvent> pHLT;
    iEvent.getByLabel(triggerEventTag_, pHLT);

    const Int_t nF(pHLT->sizeFilters());
    
    std::vector<Int_t> filterInd;

    for (Int_t itrig =0; itrig < (Int_t)vHltPathFilter_.size(); ++itrig ) {    
        std::cout << "working on #" << itrig << std::endl; std::cout << "  ---> " << vHltPathFilter_[itrig] << std::endl;        
        filterInd.push_back( pHLT->filterIndex(vHltPathFilter_[itrig]) );
    }

    Bool_t finalpathfound = false;

    for (Int_t itrig=0; itrig < (Int_t)filterInd.size(); ++itrig ) {
        std::cout << "working on #" << itrig << std::endl; std::cout << "  ---> " << filterInd[itrig] << std::endl;
        if ( nF != filterInd[itrig] ) {
            finalpathfound = true;
            break;
        }
    }

    if ( ! finalpathfound ) {
        std::cout << "No HLT Filter was not found in this event..." << std::endl;
        if ( useTriggerInfo_ ) {
            return false;    // RETURN if event fails the trigger
        }
    }

    const trigger::TriggerObjectCollection& TOC(pHLT->getObjects());


    /***    ET CUT: At least one electron in the event with ET > ETCut_     ***/

    //  Electron Collection
    edm::Handle<pat::ElectronCollection> patElectron;
    iEvent.getByLabel(electronCollectionTag_, patElectron);

    if ( ! patElectron.isValid() ) {    std::cout << "No electrons found in this event with tag " << electronCollectionTag_  << std::endl;
        return false; // RETURN if no elecs in the event
    }

    const pat::ElectronCollection *pElecs = patElectron.product();

    // check how many electrons there are in the event
    const Int_t Nelecs = pElecs->size();

    if ( Nelecs <= 1 ) {    std::cout << "No more than 1 electrons found in this event" << std::endl;
        return false; // RETURN if less than 2 elecs in the event
    }

    //  Order your electrons: first the ones with the higher ET
    Int_t  counter = 0;
    std::vector<Int_t> indices;
    std::vector<Double_t> ETs;
    pat::ElectronCollection myElectrons;

    for (pat::ElectronCollection::const_iterator elec = pElecs->begin(); elec != pElecs->end(); ++elec) {  //  the definition of  the electron ET is wrt Gsf track eta
        Double_t sc_et = elec->caloEnergy()/TMath::CosH(elec->gsfTrack()->eta());
        indices.push_back(counter);
        ETs.push_back(sc_et);
        myElectrons.push_back(*elec);
        ++counter;
    }

    const Int_t  event_elec_number = (Int_t)indices.size();

    if ( event_elec_number <= 1 ) {  std::cout << "No more than 1 electrons in fiducial were found" << std::endl;
        return false; // RETURN if no more than 1 electron in fiducial
    }

    //  Memory allocation (must be released every time we return back.
    Int_t *sorted = new Int_t[event_elec_number];
    Double_t *et = new Double_t[event_elec_number];

    for (Int_t i=0; i<event_elec_number; ++i ) {
        et[i] = ETs[i];
    }

    // array sorted now has the indices of the highest ET electrons
    TMath::Sort(event_elec_number, et, sorted, true);
    //
    // if the 2 highest electrons in the event has ET < ETCut_ return
    Int_t max_et_index1 = sorted[0];
    Int_t max_et_index2 = sorted[1];

    if ( ( ETs[max_et_index1] < ETCut_ ) || ( ETs[max_et_index2] < ETCut_ ) ) {
        delete [] sorted;
        delete [] et;
        return false; // RETURN: demand the highest ET electrons to have ET > ETcut
    }

    // my electrons now:
    pat::Electron maxETelec1 = myElectrons[max_et_index1];
    pat::Electron maxETelec2 = myElectrons[max_et_index2];

    // demand that they are in fiducial:
    if ( ! isInFiducial(maxETelec1.caloPosition().eta()) ) {
        delete [] sorted;
        delete [] et;
        return false; // RETURN highest ET electron is not in fiducial
    }

    if ( ! isInFiducial(maxETelec2.caloPosition().eta()) ) {
        delete [] sorted;
        delete [] et;
        return false; // RETURN 2nd highest ET electron is not in fiducial
    }

    // demand that they are ecal driven
    if ( useEcalDrivenElectrons_ ) {
        if ( ( ! maxETelec1.ecalDrivenSeed() ) || ( ! maxETelec2.ecalDrivenSeed() ) ) {
            delete [] sorted;
            delete [] et;
            return false; // RETURN At least one high ET electron is not ecal driven
        }
    }

    // spike rejection;
    if ( useSpikeRejection_ && maxETelec1.isEB() ) {

        edm::Handle<EcalRecHitCollection> recHits;

//         if ( maxETelec1.isEB() ) {
//             iEvent.getByLabel(ebRecHits_, recHits);
//         }
//         else {
//             iEvent.getByLabel(eeRecHits_, recHits);
//         }

        iEvent.getByLabel(ebRecHits_, recHits);

        const EcalRecHitCollection *myRecHits = recHits.product();
        const DetId seedId = maxETelec1.superCluster()->seed()->seed();

        EcalSeverityLevelAlgo severity;
        Double_t swissCross = severity.swissCross(seedId, *myRecHits);

        if ( swissCross > spikeCleaningSwissCrossCut_ ) {
            delete [] sorted;
            delete [] et;
            return false; // RETURN highest ET electron is a spike
        }
    }

    if ( useSpikeRejection_ && maxETelec2.isEB() ) {

        edm::Handle<EcalRecHitCollection> recHits;

//         if ( maxETelec2.isEB())  {
//             iEvent.getByLabel(ebRecHits_, recHits);
//         }
//         else    {
//             iEvent.getByLabel(eeRecHits_, recHits);
//         }

        iEvent.getByLabel(ebRecHits_, recHits);

        const EcalRecHitCollection *myRecHits = recHits.product();
        const DetId seedId = maxETelec2.superCluster()->seed()->seed();

        EcalSeverityLevelAlgo severity;
        Double_t swissCross = severity.swissCross(seedId, *myRecHits);

        if ( swissCross > spikeCleaningSwissCrossCut_ ) {
            delete [] sorted;
            delete [] et;
            return false; // RETURN 2nd highest ET electron is a spike
        }
    }


    //  Special pre-selection requirements (hit pattern and conversion rejection)

    if ( useValidFirstPXBHit1_ || calculateValidFirstPXBHit1_ ) {

        Bool_t fail = ( ! maxETelec1.gsfTrack()->hitPattern().hasValidHitInFirstPixelBarrel() ) ;

        if ( useValidFirstPXBHit1_ && fail ) {    std::cout << "Filter: there is no valid hit for electron #1 in 1st layer PXB" << std::endl;
            delete [] sorted;
            delete [] et;
            return false;
        }

        if ( calculateValidFirstPXBHit1_ ) {

            std::string vfpx("PassValidFirstPXBHit");

            if ( fail ) {
                maxETelec1.addUserInt(vfpx,0);
            }
            else {
                maxETelec1.addUserInt(vfpx,1);
            }

        }

    }

    if ( useValidFirstPXBHit2_ || calculateValidFirstPXBHit2_ ) {

        Bool_t fail = ( ! maxETelec2.gsfTrack()->hitPattern().hasValidHitInFirstPixelBarrel() );

        if ( useValidFirstPXBHit2_ && fail ) {   std::cout << "Filter: there is no valid hit for electron #1 in 1st layer PXB" << std::endl;
            delete [] sorted;
            delete [] et;
            return false;
        }

        if ( calculateValidFirstPXBHit2_ ) {

            std::string vfpx("PassValidFirstPXBHit");

            if ( fail ) {
                maxETelec2.addUserInt(vfpx,0);
            }
            else {
                maxETelec2.addUserInt(vfpx,1);
            }

        }

    }

    if ( useExpectedMissingHits1_ || calculateExpectedMissingHits1_ ) {

        Int_t numberOfInnerHits = (Int_t)( maxETelec1.gsfTrack()->trackerExpectedHitsInner().numberOfHits() );

        if ( ( numberOfInnerHits > maxNumberOfExpectedMissingHits1_ ) && useExpectedMissingHits1_ ) {
            delete [] sorted;
            delete [] et;
            return false;
        }

        if ( calculateExpectedMissingHits1_ ) {
            maxETelec1.addUserInt("NumberOfExpectedMissingHits",numberOfInnerHits);
        }

    }

    if ( useExpectedMissingHits2_ || calculateExpectedMissingHits2_ ) {

        Int_t numberOfInnerHits = (Int_t)( maxETelec2.gsfTrack()->trackerExpectedHitsInner().numberOfHits() );

        if ( ( numberOfInnerHits > maxNumberOfExpectedMissingHits2_ ) && useExpectedMissingHits2_ ) {
            delete [] sorted;
            delete [] et;
            return false;
        }

        if ( calculateExpectedMissingHits2_ ) {
            maxETelec2.addUserInt("NumberOfExpectedMissingHits",numberOfInnerHits);
        }
    }

    if ( useConversionRejection1_ ) {
        //  Use Conversion Rejection as it is implemented by the EGamma Group
        Double_t dist = maxETelec1.convDist();
        Double_t dcot = maxETelec1.convDcot();

        Bool_t isConv = ( ( TMath::Abs(dist) < dist1_ ) && ( TMath::Abs(dcot) < dcot1_ ) ) ;

        std::cout << "Filter: for electron #1 the conversion says " << isConv << std::endl;

        if ( isConv ) {
            delete [] sorted;
            delete [] et;
            return false;	// RETURN: it is conversion 
        }
    }
    
    if ( useConversionRejection2_ ) {
        //  Use Conversion Rejection as it is implemented by the EGamma Group
        Double_t dist = maxETelec2.convDist();
        Double_t dcot = maxETelec2.convDcot();

        Bool_t isConv = ( ( TMath::Abs(dist) < dist2_ ) && ( TMath::Abs(dcot) < dcot2_ ) ) ;

        std::cout << "Filter: for electron #2 the conversion says " << isConv << std::endl;

        if ( isConv ) {
            delete [] sorted;
            delete [] et;
            return false;	// RETURN: it is conversion 
        }
    }


    /***    TRIGGER MATCHING - Event should match the trigger     ***/
	 
	std::cout << "HLT matching starts" << std::endl;
	
	if ( electronMatched2HLT_ && useTriggerInfo_ ) {
	
	    Double_t matched_dr_distance1 = 999999.;
	    Int_t trigger_int_probe1 = 0;
	
	    Double_t matched_dr_distance2 = 999999.;
	    Int_t trigger_int_probe2 = 0;
	
	    if ( finalpathfound ) {
	
	        for (Int_t itrig=0; itrig < (Int_t)filterInd.size(); ++itrig ) {
	
	            if ( filterInd[itrig] == nF ) {
	                continue;
	            }
	
	            std::cout << "working on #" << itrig << std::endl;
	            std::cout << "  ---> " << filterInd[itrig] << std::endl;
	
	            const trigger::Keys& KEYS(pHLT->filterKeys(filterInd[itrig]));
	            const Int_t nK(KEYS.size());
	
	            std::cout << "Found trig objects #" << nK << std::endl;
	
	            for (Int_t iTrig = 0; iTrig <nK; ++iTrig ) {
	
	                const trigger::TriggerObject& TO(TOC[KEYS[iTrig]]);
	
	                if ( vUseHltObjectETCut_[itrig] == 1 ) {
	                    if ( TO.et() < vHltObjectETCut_[itrig] ) {
	                        continue;
	                    }
	                }
	
	                Double_t dr_ele_HLT1 = reco::deltaR(maxETelec1.superCluster()->eta(),maxETelec1.superCluster()->phi(),TO.eta(),TO.phi());
	                Double_t dr_ele_HLT2 = reco::deltaR(maxETelec2.superCluster()->eta(),maxETelec2.superCluster()->phi(),TO.eta(),TO.phi());
	
	                //std::cout << "-->found dr=" << dr_ele_HLT << std::endl;
	
	                if ( TMath::Abs(dr_ele_HLT1) < matched_dr_distance1 ) {
	                    matched_dr_distance1 = dr_ele_HLT1;
	                }
	
	                if ( TMath::Abs(dr_ele_HLT2) < matched_dr_distance2 ) {
	                    matched_dr_distance2 = dr_ele_HLT2;
	                }
	            }
	        }
	
	        if ( matched_dr_distance1 < electronMatched2HLT_DR_ ) {
	            ++trigger_int_probe1;
	        }
	
	        if ( matched_dr_distance2 < electronMatched2HLT_DR_ ) {
	            ++trigger_int_probe2;
	        }
	
	        if ( ( trigger_int_probe1 == 0 ) && ( trigger_int_probe2 == 0 ) ) {
	            std::cout << "No electron could be matched to an HLT object with " << std::endl;
	
	            delete [] sorted;
	            delete [] et;
	
	            return false; // RETURN: electron is not matched to an HLT object
	        }
	
	        maxETelec1.addUserFloat("HLTMatchingDR", Float_t(matched_dr_distance1));
	        maxETelec2.addUserFloat("HLTMatchingDR", Float_t(matched_dr_distance2));
	
	    }
	    else {  //std::cout << "Electron filter not found - should not be like that... " << std::endl;
	
	        delete [] sorted;
	        delete [] et;
	
	        return false; // RETURN: electron is not matched to an HLT object
	    }
	}
	
	std::cout << "HLT matching has finished" << std::endl;

    // ___________________________________________________________________
    //

    //  Add information of whether the event passes the following sets of triggers.

    if ( HLTResults.isValid() ) {

        const  std::string process = triggerCollectionTag_.process();

        Int_t triggerDecision = 0;
        
        const edm::TriggerNames & triggerNames = iEvent.triggerNames(*HLTResults);
            
        UInt_t trigger_size = HLTResults->size();

        for ( Int_t i=0; i<(Int_t)vHltPath_.size(); ++i ) {
        
            UInt_t trigger_position = triggerNames.triggerIndex(vHltPath_[i]);
            
            Int_t  passTrigger = 0;

            if ( trigger_position < trigger_size ) {
                passTrigger = (Int_t)HLTResults->accept(trigger_position);
            }

            if ( passTrigger > 0 ) {
                if ( i >= (Int_t)vHltPathFilter_.size() ) {
                    triggerDecision += (Int_t)(TMath::Power(2,i));
                }
                else {
                    const Int_t myfilterInd = pHLT->filterIndex(vHltPathFilter_[i]);
                    if ( myfilterInd != nF ) {
                        triggerDecision += (Int_t)(TMath::Power(2,i));
                    }
                }
            }
        }

        // add the info in the maxETelec1 and maxETelec2
        maxETelec1.addUserInt("triggerDecision",triggerDecision);
        maxETelec2.addUserInt("triggerDecision",triggerDecision);
    }

    // ___________________________________________________________________
    //

    // MET Collection
    edm::Handle<pat::METCollection> patMET;
    iEvent.getByLabel(metCollectionTag_,   patMET);

    edm::Handle<pat::METCollection> patpfMET;
    iEvent.getByLabel(pfMetCollectionTag_, patpfMET);

    edm::Handle<pat::METCollection> pattcMET;
    iEvent.getByLabel(tcMetCollectionTag_, pattcMET);

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
    //
    if ( metEt < METCut_ ) {    std::cout << "MET is " << metEt << std::endl;

        delete [] sorted;
        delete [] et;

        return false;  // RETURN if MET is < Metcut
    }


    // if you have indeed reached this point then you have a zeeCandidate!!!

    pat::CompositeCandidate zeeCandidate;

    zeeCandidate.addDaughter(maxETelec1, "electron1");
    zeeCandidate.addDaughter(maxETelec2, "electron2");

    zeeCandidate.addDaughter(theMET, "met");
    zeeCandidate.addDaughter(thePfMET, "pfmet");
    zeeCandidate.addDaughter(theTcMET, "tcmet");

    auto_ptr<pat::CompositeCandidateCollection>selectedZeeCandidates(new pat::CompositeCandidateCollection);

    selectedZeeCandidates->push_back(zeeCandidate);

    iEvent.put(selectedZeeCandidates, "selectedZeeCandidates");

    // release your memory
    delete [] sorted;
    delete [] et;

    std::cout << "Run = "   << iEvent.run() << " " 
              << "Lumi = "  << (Int_t)iEvent.luminosityBlock() << " " 
              << "Event = " << iEvent.eventAuxiliary().event() << " " 
              << "FILTER-MSG: Event Accepted for Z Candidate"
              << std::endl;

    return true;

}

// ------------ method called once each job just after ending the event loop  -
void ZeeCandidateFilter::endJob() {}

Bool_t ZeeCandidateFilter::isInFiducial(Double_t eta)
{
    if ( TMath::Abs(eta) < BarrelMaxEta_ ) {
        return true;
    }
    else if ( ( TMath::Abs(eta) < EndCapMaxEta_ ) && ( TMath::Abs(eta) > EndCapMinEta_ ) ) {
        return true;
    }

    return false;

}

//define this as a plug-in
DEFINE_FWK_MODULE(ZeeCandidateFilter);

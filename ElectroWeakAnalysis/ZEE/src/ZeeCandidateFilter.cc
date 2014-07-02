// -*- C++ -*-
//
// Package:    ZeeCandidateFilter
// Class:      ZeeCandidateFilter
//
/**\class ZeeCandidateFilter ZeeCandidateFilter.cc EWKSoftware/EDMTupleSkimmerFilter/src/ZeeCandidateFilter.cc

 Description: <one line class summary>

 Implementation:

    This class contains a filter that searches the event and finds whether it fulfills the Z Candidate Criteria.
    If it fullfills them it creates a ZeeCandidate and stores it in the event.

    Definition of the Zee Caldidate:
    * event that passes the trigger
    * has 2 Gsf electrons in fiducial with ET greater than a (configurable) threshold
    * at least one of them matched to an HLT Object (configurable) with DR < (configurable)

 Changes Log:

 12Feb09  First Release of the code for CMSSW_2_2_X

 17Sep09  First Release for CMSSW_3_1_X

 09Dec09  Option to ignore trigger

 25Feb10  Added options to use Conversion Rejection, Expected missing hits and valid hit at first PXB

          Added option to calculate these criteria and store them in the pat electron object this is done by setting in the configuration the flags

                calculateValidFirstPXBHit = true
                calculateConversionRejection = true
                calculateExpectedMissinghits = true

          Then the code calculates them and you can access all these from pat::Electron

                myElec.userInt("PassValidFirstPXBHit")      0 fail, 1 passes
                myElec.userInt("PassConversionRejection")   0 fail, 1 passes
                myElec.userInt("NumberOfExpectedMissingHits") the number of lost hits


 28May10  Implementation of Spring10 selections
 Contact:
 Stilianos Kesisoglou - Institute of Nuclear Physics
 NCSR Demokritos
// Original Author:  Nikolaos Rompotis

 Nikolaos.Rompotis@Cern.ch
 Imperial College London

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
//#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalTools.h"
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

    virtual Bool_t filter(edm::Event&, const edm::EventSetup&) override;

    virtual void endJob() override ;

    Bool_t isInFiducial(Double_t eta);

    //Bool_t passEleIDCuts(pat::Electron *ele);

    //  -----   Data Members    -----

    Double_t                    ETCut_                                      ;
    Double_t                    METCut_                                     ;

    Bool_t                      useEcalDrivenElectrons_                     ;

    /*  Electron 1  */
    Bool_t                      useValidFirstPXBHit1_                       ;
    Bool_t                      calculateValidFirstPXBHit1_                 ;
    Bool_t                      useConversionRejection1_                    ;
    Bool_t                      calculateConversionRejection1_              ;
    Bool_t                      useExpectedMissingHits1_                    ;
    Bool_t                      calculateExpectedMissingHits1_              ;
    Int_t                       maxNumberOfExpectedMissingHits1_            ;

    /*  Electron 2  */
    Bool_t                      useValidFirstPXBHit2_                       ;
    Bool_t                      calculateValidFirstPXBHit2_                 ;
    Bool_t                      useConversionRejection2_                    ;
    Bool_t                      calculateConversionRejection2_              ;
    Bool_t                      useExpectedMissingHits2_                    ;
    Bool_t                      calculateExpectedMissingHits2_              ;
    Int_t                       maxNumberOfExpectedMissingHits2_            ;

    /*  Electron 1  */
    Double_t                    dist1_                                      ;
    Double_t                    dcot1_                                      ;

    /*  Electron 2  */
    Double_t                    dist2_                                      ;
    Double_t                    dcot2_                                      ;

    Bool_t                      dataMagneticFieldSetUp_                     ;

    edm::InputTag               dcsTag_                                     ;
    edm::EDGetTokenT<DcsStatusCollection>               dcsToken_                                     ;
    edm::EDGetTokenT<reco::TrackCollection> tracksToken_;

    Double_t                    BarrelMaxEta_                               ;
    Double_t                    EndCapMaxEta_                               ;
    Double_t                    EndCapMinEta_                               ;

    std::string                 hltpath_                                    ;
    edm::InputTag               triggerCollectionTag_                       ;
    edm::EDGetTokenT<edm::TriggerResults>               triggerCollectionToken_                       ;
    edm::EDGetTokenT<trigger::TriggerEvent>               triggerEventToken_                            ;
    edm::InputTag               hltpathFilter_                              ;
    Bool_t                      useHLTObjectETCut_                          ;

    Double_t                    hltObjectETCut_                             ;

    Bool_t                      useExtraTrigger_                            ;

    std::vector<std::string>    vHltpathExtra_                              ;
    std::vector<edm::InputTag>  vHltpathFilterExtra_                        ;

    Bool_t                      useTriggerInfo_                             ;
    Bool_t                      electronMatched2HLT_                        ;
    Double_t                    electronMatched2HLT_DR_                     ;

    edm::InputTag               electronCollectionTag_                      ;
    edm::EDGetTokenT<pat::ElectronCollection>               electronCollectionToken_                      ;

    edm::EDGetTokenT<pat::METCollection>               metCollectionToken_                           ;
    edm::EDGetTokenT<pat::METCollection>               pfMetCollectionToken_                         ;
    edm::EDGetTokenT<pat::METCollection>               tcMetCollectionToken_                         ;

    edm::EDGetTokenT< std::vector<reco::Vertex> >               PrimaryVerticesCollectionToken_                  ;

    edm::EDGetTokenT<EcalRecHitCollection>               ebRecHitsToken_                                  ;
    edm::EDGetTokenT<EcalRecHitCollection>               eeRecHitsToken_                                  ;

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
    //
    //-------------------------------------//
    //         INITIALIZATION              //
    //-------------------------------------//
    //


    //  Cuts
    //  ----
    ETCut_  = iConfig.getUntrackedParameter<Double_t>("ETCut");
    METCut_ = iConfig.getUntrackedParameter<Double_t>("METCut");

    useEcalDrivenElectrons_ = iConfig.getUntrackedParameter<Bool_t>("useEcalDrivenElectrons", false);
    //--------------------------------------------------------------------------------------------------------------------


    //  Preselection Criteria: Hit Pattern
    //  ----------------------------------
    //
    /*  Electron 1  */
    useValidFirstPXBHit1_             =  iConfig.getUntrackedParameter<Bool_t>("useValidFirstPXBHit1",false);
    calculateValidFirstPXBHit1_       =  iConfig.getUntrackedParameter<Bool_t>("calculateValidFirstPXBHit1",false);
    useConversionRejection1_          =  iConfig.getUntrackedParameter<Bool_t>("useConversionRejection1",false);
    calculateConversionRejection1_    =  iConfig.getUntrackedParameter<Bool_t>("calculateConversionRejection1",false);
    useExpectedMissingHits1_          =  iConfig.getUntrackedParameter<Bool_t>("useExpectedMissingHits1",false);
    calculateExpectedMissingHits1_    =  iConfig.getUntrackedParameter<Bool_t>("calculateExpectedMissingHits1",false);
    maxNumberOfExpectedMissingHits1_  =  iConfig.getUntrackedParameter<Int_t>("maxNumberOfExpectedMissingHits1",1);
    //
    /*  Electron 2  */
    useValidFirstPXBHit2_             =  iConfig.getUntrackedParameter<Bool_t>("useValidFirstPXBHit2",false);
    calculateValidFirstPXBHit2_       =  iConfig.getUntrackedParameter<Bool_t>("calculateValidFirstPXBHit2",false);
    useConversionRejection2_          =  iConfig.getUntrackedParameter<Bool_t>("useConversionRejection2",false);
    calculateConversionRejection2_    =  iConfig.getUntrackedParameter<Bool_t>("calculateConversionRejection2",false);
    useExpectedMissingHits2_          =  iConfig.getUntrackedParameter<Bool_t>("useExpectedMissingHits2",false);
    calculateExpectedMissingHits2_    =  iConfig.getUntrackedParameter<Bool_t>("calculateExpectedMissingHits2",false);
    maxNumberOfExpectedMissingHits2_  =  iConfig.getUntrackedParameter<Int_t>("maxNumberOfExpectedMissingHits2",1);
    //--------------------------------------------------------------------------------------------------------------------


    //  Conversion Rejection Variables
    //  ------------------------------
    //
    /*  Electron 1  */
    Double_t dist1_D  = 0.02 ;
    Double_t dcot1_D  = 0.02 ;
    //
    dist1_ = iConfig.getUntrackedParameter<Double_t>("conversionRejectionDist1", dist1_D);
    dcot1_ = iConfig.getUntrackedParameter<Double_t>("conversionRejectionDcot1", dcot1_D);
    //
    /*  Electron 2  */
    Double_t dist2_D  = 0.02 ;
    Double_t dcot2_D  = 0.02 ;
    //
    dist2_ = iConfig.getUntrackedParameter<Double_t>("conversionRejectionDist2", dist2_D);
    dcot2_ = iConfig.getUntrackedParameter<Double_t>("conversionRejectionDcot2", dcot2_D);
    //--------------------------------------------------------------------------------------------------------------------


    //  Magnetic Field
    //  --------------
    //
    dataMagneticFieldSetUp_ = iConfig.getUntrackedParameter<Bool_t>("dataMagneticFieldSetUp",false);

    if ( dataMagneticFieldSetUp_ ) {
        dcsTag_ = iConfig.getUntrackedParameter<edm::InputTag>("dcsTag");
        dcsToken_ = mayConsume<DcsStatusCollection>(dcsTag_);
    }
    tracksToken_ = mayConsume<reco::TrackCollection>(edm::InputTag("generalTracks"));
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
    hltpath_              = iConfig.getUntrackedParameter<std::string>("hltpath");
    triggerCollectionTag_ = iConfig.getUntrackedParameter<edm::InputTag>("triggerCollectionTag");
    triggerCollectionToken_ = consumes<edm::TriggerResults>(triggerCollectionTag_);
    triggerEventToken_      = consumes<trigger::TriggerEvent>(iConfig.getUntrackedParameter<edm::InputTag>("triggerEventTag"));
    hltpathFilter_        = iConfig.getUntrackedParameter<edm::InputTag>("hltpathFilter");
    useHLTObjectETCut_    = iConfig.getUntrackedParameter<Bool_t>("useHLTObjectETCut", false);

    if ( useHLTObjectETCut_ ) {
        hltObjectETCut_     = iConfig.getUntrackedParameter<Double_t>("hltObjectETCut");
    }

    //  Dirty way to add a second trigger with OR, to be done properly in the next tag
    useExtraTrigger_ = iConfig.getUntrackedParameter<Bool_t>("useExtraTrigger");

    if ( useExtraTrigger_ ) {

        vHltpathExtra_       = iConfig.getUntrackedParameter< std::vector<std::string> >("vHltpathExtra");
        vHltpathFilterExtra_ = iConfig.getUntrackedParameter< std::vector<edm::InputTag> >("vHltpathFilterExtra");

        if ( Int_t(vHltpathExtra_.size()) != Int_t(vHltpathFilterExtra_.size()) ) {
            std::cout << "ZeeCandidateFilter: ERROR IN Configuration: vHltpathExtra and vHltpathFilterExtra" << " should have the same dimensions " << std::endl;
        }
    }
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
    electronCollectionToken_ = consumes<pat::ElectronCollection>(electronCollectionTag_);

    metCollectionToken_   = consumes<pat::METCollection>(iConfig.getUntrackedParameter<edm::InputTag>("metCollectionTag"));
    pfMetCollectionToken_ = consumes<pat::METCollection>(iConfig.getUntrackedParameter<edm::InputTag>("pfMetCollectionTag"));
    tcMetCollectionToken_ = consumes<pat::METCollection>(iConfig.getUntrackedParameter<edm::InputTag>("tcMetCollectionTag"));

    PrimaryVerticesCollectionToken_ = consumes< std::vector<reco::Vertex> >(iConfig.getUntrackedParameter<edm::InputTag>("PrimaryVerticesCollection"));

    ebRecHitsToken_ = mayConsume<EcalRecHitCollection>(iConfig.getUntrackedParameter<edm::InputTag>("ebRecHits"));
    eeRecHitsToken_ = mayConsume<EcalRecHitCollection>(iConfig.getUntrackedParameter<edm::InputTag>("eeRecHits"));
    //--------------------------------------------------------------------------------------------------------------------


    //  Spike Cleaning
    //  --------------
    //
    useSpikeRejection_ = iConfig.getUntrackedParameter<Bool_t>("useSpikeRejection");

    if ( useSpikeRejection_ ) {
        spikeCleaningSwissCrossCut_ = iConfig.getUntrackedParameter<Double_t>("spikeCleaningSwissCrossCut");
    }
    //--------------------------------------------------------------------------------------------------------------------


    //
    //-------------------------------------//
    //         SUMMARY PRINTOUT            //
    //-------------------------------------//
    //

    std::cout << "ZeeCandidateFilter: Running Zee Filter..." << std::endl;

    if ( useTriggerInfo_ ) {
        std::cout << "ZeeCandidateFilter: HLT Path   " << hltpath_       << std::endl;
        std::cout << "ZeeCandidateFilter: HLT Filter " << hltpathFilter_ << std::endl;

        if ( useExtraTrigger_ ) {
            for (Int_t itrig=0; itrig < (Int_t)vHltpathExtra_.size(); ++itrig ) {

                std::cout << "ZeeCandidateFilter: OR " << vHltpathExtra_[itrig] << " with filter: " << vHltpathFilterExtra_[itrig] << std::endl;
            }
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

    if ( calculateConversionRejection1_ ) {
        std::cout << "ZeeCandidateFilter: EGAMMA Conversion Rejection criteria for electron candidate #1 will be calculated and stored: you can access them later by demanding for a successful electron myElec.userInt(\"PassConversionRejection\")==1" << std::endl;
    }

    if ( calculateConversionRejection2_ ) {
        std::cout << "ZeeCandidateFilter: EGAMMA Conversion Rejection criteria for electron candidate #2 will be calculated and stored: you can access them later by demanding for a successful electron myElec.userInt(\"PassConversionRejection\")==1" << std::endl;
    }

    if ( dataMagneticFieldSetUp_ ) {
        std::cout << "ZeeCandidateFilter: Data Configuration for Magnetic Field DCS tag " << dcsTag_  << std::endl;
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
    iEvent.getByToken(triggerCollectionToken_, HLTResults);

    Int_t passTrigger = 0;

    if ( HLTResults.isValid() ) {

        const edm::TriggerNames & triggerNames = iEvent.triggerNames(*HLTResults);

        UInt_t trigger_size     = HLTResults->size();
        UInt_t trigger_position = triggerNames.triggerIndex(hltpath_);
        UInt_t trigger_position_extra;

        if ( trigger_position < trigger_size ) {
            passTrigger = (Int_t)HLTResults->accept(trigger_position);
        }

        //  Tested TriggerPath firing results printout
        std::cout << "SK_HLT_INFO"
                  << " | " << "trigger_size = "         << trigger_size
                  << " | " << "hltpath_ = "             << hltpath_
                  << " | " << "trigger_position = "     << trigger_position
                  << " | " << "passTrigger = "          << passTrigger
        << std::endl;

        if ( useExtraTrigger_ && passTrigger==0 ) {
            for (Int_t itrig=0; itrig < (Int_t)vHltpathExtra_.size(); ++itrig ) {
                trigger_position_extra = triggerNames.triggerIndex(vHltpathExtra_[itrig]);

                if ( trigger_position_extra < trigger_size ) {
                    passTrigger = (Int_t)HLTResults->accept(trigger_position_extra);
                }

                //  Tested TriggerPath firing results printout
                std::cout << "SK_HLT_INFO"
                          << " | " << "vHltpathExtra_[" << itrig << "] = "      << vHltpathExtra_[itrig]
                          << " | " << "trigger_position_extra = "               << trigger_position_extra
                          << " | " << "passTrigger = "                          << passTrigger
                          << " | " << "vHltpathExtra_.size() = "                << vHltpathExtra_.size()
                << std::endl;

                if ( passTrigger > 0 ) { break ; }

            }   //  for Loop

        }   // if ( useExtraTrigger_ && passTrigger==0 )

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
    iEvent.getByToken(triggerEventToken_, pHLT);

    const Int_t nF(pHLT->sizeFilters());
    const Int_t filterInd = pHLT->filterIndex(hltpathFilter_);

    std::vector<Int_t> filterIndExtra;

    if ( useExtraTrigger_ ) {
        for (Int_t itrig =0; itrig < (Int_t)vHltpathFilterExtra_.size(); ++itrig ) {     std::cout << "working on #" << itrig << std::endl; std::cout << "  ---> " << vHltpathFilterExtra_[itrig] << std::endl;
            filterIndExtra.push_back( pHLT->filterIndex(vHltpathFilterExtra_[itrig]) );
        }
    }

    Bool_t finalpathfound = false;

    if ( nF != filterInd ) {
        finalpathfound = true;
    }
    else {
        for (Int_t itrig=0; itrig < (Int_t)filterIndExtra.size(); ++itrig ) {    std::cout << "working on #" << itrig << std::endl; std::cout << "  ---> " << filterIndExtra[itrig] << std::endl;
            if ( nF != filterIndExtra[itrig] ) {
                finalpathfound = true;
                break;
            }
        }
    }

    if ( ! finalpathfound ) {   std::cout << "No HLT Filter was not found in this event..." << std::endl;
        if ( useTriggerInfo_ ) {
            return false;    // RETURN if event fails the trigger
        }
    }

    const trigger::TriggerObjectCollection& TOC(pHLT->getObjects());

    /***    ET CUT: At least one electron in the event with ET > ETCut_     ***/

    //  Electron Collection
    edm::Handle<pat::ElectronCollection> patElectron;
    iEvent.getByToken(electronCollectionToken_, patElectron);

    if ( ! patElectron.isValid() ) {    std::cout << "No electrons found in this event with tag " << electronCollectionTag_  << std::endl;
        return false; // RETURN if no elecs in the event
    }

    const pat::ElectronCollection *pElecs = patElectron.product();

//     // MET Collection                                            ->  relocated block bellow
//     edm::Handle<pat::METCollection> patMET;
//     iEvent.getByToken(metCollectionToken_,   patMET);
//
//     edm::Handle<pat::METCollection> patpfMET;
//     iEvent.getByToken(pfMetCollectionToken_, patpfMET);
//
//     edm::Handle<pat::METCollection> pattcMET;
//     iEvent.getByToken(tcMetCollectionToken_, pattcMET);

    //
    // Note: best to do Duplicate removal here, since the current
    // implementation does not remove triplicates
    // duplicate removal is on at PAT, but does it remove triplicates?
    //

//     pat::ElectronCollection::const_iterator elec;   //  relocated bellow

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
//             iEvent.getByToken(ebRecHitsToken_, recHits);
//         }
//         else {
//             iEvent.getByToken(eeRecHitsToken_, recHits);
//         }

        iEvent.getByToken(ebRecHitsToken_, recHits);

        const EcalRecHitCollection *myRecHits = recHits.product();
        const DetId seedId = maxETelec1.superCluster()->seed()->seed();

        Double_t swissCross = EcalTools::swissCross(seedId, *myRecHits,0.);

        if ( swissCross > spikeCleaningSwissCrossCut_ ) {
            delete [] sorted;
            delete [] et;
            return false; // RETURN highest ET electron is a spike
        }
    }

    if ( useSpikeRejection_ && maxETelec2.isEB() ) {

        edm::Handle<EcalRecHitCollection> recHits;

//         if ( maxETelec2.isEB())  {
//             iEvent.getByToken(ebRecHitsToken_, recHits);
//         }
//         else    {
//             iEvent.getByToken(eeRecHitsToken_, recHits);
//         }

        iEvent.getByToken(ebRecHitsToken_, recHits);

        const EcalRecHitCollection *myRecHits = recHits.product();
        const DetId seedId = maxETelec2.superCluster()->seed()->seed();

        Double_t swissCross = EcalTools::swissCross(seedId, *myRecHits,0.);

        if ( swissCross > spikeCleaningSwissCrossCut_ ) {
            delete [] sorted;
            delete [] et;
            return false; // RETURN 2nd highest ET electron is a spike
        }
    }

    // add the primary vtx information in the electron:
    edm::Handle< std::vector<reco::Vertex> > pVtx;
    iEvent.getByToken(PrimaryVerticesCollectionToken_, pVtx);

    const std::vector<reco::Vertex> Vtx = *(pVtx.product());

    Double_t pv_x = -999999.;
    Double_t pv_y = -999999.;
    Double_t pv_z = -999999.;

    Double_t ele_tip_pv1 = -999999.;
    Double_t ele_tip_pv2 = -999999.;

    if ( Vtx.size() >=1 ) {
        pv_x = Vtx[0].position().x();
        pv_y = Vtx[0].position().y();
        pv_z = Vtx[0].position().z();
        ele_tip_pv1 = (-1.0) * ( maxETelec1.gsfTrack()->dxy(Vtx[0].position()) ) ;
        ele_tip_pv2 = (-1.0) * ( maxETelec2.gsfTrack()->dxy(Vtx[0].position()) ) ;
    }

    maxETelec1.addUserFloat("pv_x", Float_t(pv_x));
    maxETelec1.addUserFloat("pv_x", Float_t(pv_y));
    maxETelec1.addUserFloat("pv_z", Float_t(pv_z));
    maxETelec1.addUserFloat("ele_tip_pv", Float_t(ele_tip_pv1));

    maxETelec2.addUserFloat("pv_x", Float_t(pv_x));
    maxETelec2.addUserFloat("pv_x", Float_t(pv_y));
    maxETelec2.addUserFloat("pv_z", Float_t(pv_z));
    maxETelec2.addUserFloat("ele_tip_pv", Float_t(ele_tip_pv2));

//     Double_t pv_x1 = -999999.;
//     Double_t pv_y1 = -999999.;
//     Double_t pv_z1 = -999999.;
//     Double_t ele_tip_pv1 = -999999.;
//
//     if ( Vtx.size() >=1 ) {
//         pv_x1 = Vtx[0].position().x();
//         pv_y1 = Vtx[0].position().y();
//         pv_z1 = Vtx[0].position().z();
//         ele_tip_pv1 = (-1.0) * ( maxETelec1.gsfTrack()->dxy(Vtx[0].position()) ) ;
//     }
//
//     maxETelec1.addUserFloat("pv_x", Float_t(pv_x1));
//     maxETelec1.addUserFloat("pv_x", Float_t(pv_y1));
//     maxETelec1.addUserFloat("pv_z", Float_t(pv_z1));
//     maxETelec1.addUserFloat("ele_tip_pv", Float_t(ele_tip_pv1));
//
//     edm::Handle< std::vector<reco::Vertex> > pVtx2;
//     iEvent.getByToken(PrimaryVerticesCollectionToken_, pVtx2);
//
//     const std::vector<reco::Vertex> Vtx2 = *(pVtx2.product());
//
//     Double_t pv_x2 = -999999.;
//     Double_t pv_y2 = -999999.;
//     Double_t pv_z2 = -999999.;
//     Double_t ele_tip_pv2 = -999999.;
//
//     if ( Vtx2.size() >=1 ) {
//         pv_x2 = Vtx2[0].position().x();
//         pv_y2 = Vtx2[0].position().y();
//         pv_z2 = Vtx2[0].position().z();
//         ele_tip_pv2 = -maxETelec2.gsfTrack()->dxy(Vtx2[0].position());
//     }
//
//     maxETelec2.addUserFloat("pv_x", Float_t(pv_x1));
//     maxETelec2.addUserFloat("pv_x", Float_t(pv_y1));
//     maxETelec2.addUserFloat("pv_z", Float_t(pv_z1));
//     maxETelec2.addUserFloat("ele_tip_pv", Float_t(ele_tip_pv2));


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

    if ( useConversionRejection1_ || calculateConversionRejection1_ ) {
        // use of conversion rejection as it is implemented in egamma
        // you have to get the general track collection to do that
        // WARNING! you have to supply the correct B-field in Tesla
        // the magnetic field

        Double_t bfield;

        if ( dataMagneticFieldSetUp_ ) {

            edm::Handle<DcsStatusCollection> dcsHandle;
            iEvent.getByToken(dcsToken_, dcsHandle);
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
            bfield = mField->inTesla(GlobalPoint(0.0,0.0,0.0)).z();

        }

        edm::Handle<reco::TrackCollection> ctfTracks;

        if ( iEvent.getByToken(tracksToken_, ctfTracks) ) {

            ConversionFinder convFinder;
            ConversionInfo convInfo = convFinder.getConversionInfo(maxETelec1, ctfTracks, bfield);

            Float_t dist = convInfo.dist();
            Float_t dcot = convInfo.dcot();

            Bool_t isConv = ( ( TMath::Abs(dist) < dist1_ ) && ( TMath::Abs(dcot) < dcot1_ ) ) ;

            std::cout << "Filter: for electron #1 the conversion says " << isConv << std::endl;

            if ( isConv && useConversionRejection1_ ) {
                delete [] sorted;
                delete [] et;
                return false;
            }

            if ( calculateConversionRejection1_ ) {

                maxETelec1.addUserFloat("Dist", Float_t(dist));
                maxETelec1.addUserFloat("Dcot", Float_t(dcot));

                if ( isConv ) {
                    maxETelec1.addUserInt("PassConversionRejection",0);
                }
                else {
                    maxETelec1.addUserInt("PassConversionRejection",1);
                }

            }

        }
        else {
            std::cout << "WARNING! Track Collection with input name: generalTracks was not found. Conversion Rejection for electron #1 is not going to be applied!!!" << std::endl;
        }

    }

    if ( useConversionRejection2_ || calculateConversionRejection2_ ) {
        // use of conversion rejection as it is implemented in egamma
        // you have to get the general track collection to do that
        // WARNING! you have to supply the correct B-field in Tesla
        // the magnetic field

        Double_t bfield;

        if ( dataMagneticFieldSetUp_ ) {

            edm::Handle<DcsStatusCollection> dcsHandle;
            iEvent.getByToken(dcsToken_, dcsHandle);

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
            bfield = mField->inTesla(GlobalPoint(0.0,0.0,0.0)).z();

        }

        edm::Handle<reco::TrackCollection> ctfTracks;

        if ( iEvent.getByToken(tracksToken_, ctfTracks) ) {

            ConversionFinder convFinder;
            ConversionInfo convInfo = convFinder.getConversionInfo(maxETelec2, ctfTracks, bfield);

            Float_t dist = convInfo.dist();
            Float_t dcot = convInfo.dcot();

            Bool_t isConv = ( ( TMath::Abs(dist) < dist2_ ) && ( TMath::Abs(dcot) < dcot2_ ) ) ;

            std::cout << "Filter: for electron #2 the conversion says " << isConv << std::endl;

            if ( isConv && useConversionRejection2_ ) {
                delete [] sorted;
                delete [] et;
                return false;
            }

            if ( calculateConversionRejection2_ ) {

                maxETelec2.addUserFloat("Dist", Float_t(dist));
                maxETelec2.addUserFloat("Dcot", Float_t(dcot));

                if ( isConv ) {
                    maxETelec2.addUserInt("PassConversionRejection",0);
                }
                else {
                    maxETelec2.addUserInt("PassConversionRejection",1);
                }

            }

        }
        else {
            std::cout << "WARNING! Track Collection with input name: generalTracks was not found. Conversion Rejection for electron #2 is not going to be applied!!!" << std::endl;
        }

    }

    std::cout << "HLT matching starts" << std::endl;

    if ( electronMatched2HLT_ && useTriggerInfo_ ) {

        Double_t matched_dr_distance1 = 999999.;
        Int_t trigger_int_probe1 = 0;

        Double_t matched_dr_distance2 = 999999.;
        Int_t trigger_int_probe2 = 0;

        if ( finalpathfound ) {

            if ( nF != filterInd ) {

                const trigger::Keys& KEYS(pHLT->filterKeys(filterInd));
                const Int_t nK(KEYS.size());

                std::cout << "Found trig objects #" << nK << std::endl;

                for ( Int_t iTrig = 0; iTrig < nK; ++iTrig ) {

                    const trigger::TriggerObject& TO(TOC[KEYS[iTrig]]);

                    if ( useHLTObjectETCut_ ) {
                        if ( TO.et() < hltObjectETCut_ ) {
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

            if ( useExtraTrigger_ ) {

                for (Int_t itrig=0; itrig < (Int_t) filterIndExtra.size(); ++itrig ) {

                    if ( filterIndExtra[itrig] == nF ) {
                        continue;
                    }

                    std::cout << "working on #" << itrig << std::endl; std::cout << "  ---> " << filterIndExtra[itrig] << std::endl;

                    const trigger::Keys& KEYS(pHLT->filterKeys(filterIndExtra[itrig]));
                    const Int_t nK(KEYS.size());

                    std::cout << "Found trig objects #" << nK << std::endl;

                    for (Int_t iTrig = 0; iTrig <nK; ++iTrig ) {

                        const trigger::TriggerObject& TO(TOC[KEYS[iTrig]]);

                        Double_t dr_ele_HLT1 = reco::deltaR(maxETelec1.eta(),maxETelec1.phi(),TO.eta(),TO.phi());
                        Double_t dr_ele_HLT2 = reco::deltaR(maxETelec2.eta(),maxETelec2.phi(),TO.eta(),TO.phi());

                        //std::cout << "-->found dr=" << dr_ele_HLT << std::endl;

                        if ( TMath::Abs(dr_ele_HLT1) < matched_dr_distance1 ) {
                            matched_dr_distance1 = dr_ele_HLT1;
                        }

                        if ( TMath::Abs(dr_ele_HLT2) < matched_dr_distance2 ) {
                            matched_dr_distance2 = dr_ele_HLT2;
                        }
                    }
                }
            }

            if ( matched_dr_distance1 < electronMatched2HLT_DR_ ) {
                ++trigger_int_probe1;
            }

            if ( matched_dr_distance2 < electronMatched2HLT_DR_ ) {
                ++trigger_int_probe2;
            }

            if ( ( trigger_int_probe1 == 0 ) && ( trigger_int_probe2 == 0 ) ) {    std::cout << "No electron could be matched to an HLT object with " << std::endl;

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

    // add information of whether the event passes the following sets of
    // triggers. Currently Hardwired, to be changed in the future

    if ( HLTResults.isValid() ) {

        const  std::string process = triggerCollectionTag_.process();
        //
        std::string  HLTPath[18];
        HLTPath[0 ] = "HLT_Photon10_L1R"              ;
        HLTPath[1 ] = "HLT_Photon15_L1R"              ;
        HLTPath[2 ] = "HLT_Photon20_L1R"              ;
        HLTPath[3 ] = "HLT_Photon15_TrackIso_L1R"     ;
        HLTPath[4 ] = "HLT_Photon15_LooseEcalIso_L1R" ;
        HLTPath[5 ] = "HLT_Photon30_L1R_8E29"         ;
        HLTPath[6 ] = "HLT_Photon30_L1R_8E29"         ;
        HLTPath[7 ] = "HLT_Ele10_LW_L1R"              ;
        HLTPath[8 ] = "HLT_Ele15_LW_L1R"              ;
        HLTPath[9 ] = "HLT_Ele20_LW_L1R"              ;
        HLTPath[10] = "HLT_Ele10_LW_EleId_L1R"        ;
        HLTPath[11] = "HLT_Ele15_SiStrip_L1R"         ;
        HLTPath[12] = "HLT_IsoTrackHB_8E29"           ;
        HLTPath[13] = "HLT_IsoTrackHE_8E29"           ;
        HLTPath[14] = "HLT_DiJetAve15U_8E29"          ;
        HLTPath[15] = "HLT_MET45"                     ;
        HLTPath[16] = "HLT_L1MET20"                   ;
        HLTPath[17] = "HLT_MET100"                    ;
        //
        edm::InputTag HLTFilterType[15];
        HLTFilterType[0 ]= edm::InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt10HcalIsolFilter","",process)            ;  //HLT_Photon10_L1R
        HLTFilterType[1 ]= edm::InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter" ,"",process)           ;  //HLT_Photon15_L1R
        HLTFilterType[2 ]= edm::InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt20HcalIsolFilter" ,"",process)           ;  //HLT_Photon20_L1R
        HLTFilterType[3 ]= edm::InputTag("hltL1NonIsoSinglePhotonEt15HTITrackIsolFilter","",process)                 ;  //HLT_Photon15_TrackIso_L1R
        HLTFilterType[4 ]= edm::InputTag("hltL1NonIsoSinglePhotonEt15LEIHcalIsolFilter","",process)                  ;  //HLT_Photon15_LooseEcalIso_L1R
        HLTFilterType[5 ]= edm::InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt15EtFilterESet308E29","",process)        ;  //HLT_Photon30_L1R_8E29
        HLTFilterType[6 ]= edm::InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter","",process)            ;  //HLT_Photon30_L1R_8E29
        HLTFilterType[7 ]= edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt10PixelMatchFilter","",process)      ;
        HLTFilterType[8 ]= edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt15PixelMatchFilter","",process)      ;
        HLTFilterType[9 ]= edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt15EtFilterESet20","",process)        ;
        HLTFilterType[10]= edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdDphiFilter","",process)       ;
        HLTFilterType[11]= edm::InputTag("hltL1NonIsoHLTNonIsoSingleElectronSiStripEt15PixelMatchFilter","",process) ;
        HLTFilterType[12]= edm::InputTag("hltIsolPixelTrackL3FilterHB8E29","",process)                               ;
        HLTFilterType[13]= edm::InputTag("hltIsolPixelTrackL2FilterHE8E29","",process)                               ;
        HLTFilterType[14]= edm::InputTag("hltL1sDiJetAve15U8E29","",process)                                         ;
        //
        Int_t triggerDecision = 0;
        UInt_t trigger_size = HLTResults->size();

        for (Int_t i=0; i<18; ++i ) {

            const edm::TriggerNames & triggerNames = iEvent.triggerNames(*HLTResults);
            UInt_t trigger_position = triggerNames.triggerIndex(HLTPath[i]);
            Int_t  passTrigger = 0;

            if ( trigger_position < trigger_size ) {
                passTrigger = (Int_t)HLTResults->accept(trigger_position);
            }

            if ( passTrigger > 0 ) {
                if ( i >= 15 ) {
                    triggerDecision += (Int_t)(TMath::Power(2,i));
                }
                else {
                    const Int_t myfilterInd = pHLT->filterIndex(HLTFilterType[i]);
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
    iEvent.getByToken(metCollectionToken_,   patMET);

    edm::Handle<pat::METCollection> patpfMET;
    iEvent.getByToken(pfMetCollectionToken_, patpfMET);

    edm::Handle<pat::METCollection> pattcMET;
    iEvent.getByToken(tcMetCollectionToken_, pattcMET);

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

// Bool_t ZeeCandidateFilter::passEleIDCuts(pat::Electron *ele)
// {
//     if ( ! useVetoSecondElectronID_)  return true;
//     if ( ! ele->isElectronIDAvailable(vetoSecondElectronIDType_) ) {
//         std::cout << "ZeeCandidateFilter: request ignored: 2nd electron ID type "
//                   << "not found in electron object" << std::endl;
//         return true;
//     }
//     if ( vetoSecondElectronIDSign_ == ">" ) {
//         if ( ele->electronID(vetoSecondElectronIDType_)>vetoSecondElectronIDValue_)
//             return true;
//         else return false;
//     }
//     else if ( vetoSecondElectronIDSign_ == "<" ) {
//         if ( ele->electronID(vetoSecondElectronIDType_)<vetoSecondElectronIDValue_)
//             return true;
//         else return false;
//     }
//     else {
//         if ( TMath::Abs(ele->electronID(vetoSecondElectronIDType_)-
//                        vetoSecondElectronIDValue_) < 0.1)
//             return true;
//         else return false;
//     }
// }

//define this as a plug-in
DEFINE_FWK_MODULE(ZeeCandidateFilter);

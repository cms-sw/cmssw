//L1TGlobalProducer - Emulate L1T uGT
//Author: Brian Winer  Ohio State

#ifndef L1TGLOBALPRODUCER_H
#define L1TGLOBALPRODUCER_H

// system include files
#include <string>
#include <vector>
#include<iostream>
#include<fstream>

#include <boost/cstdint.hpp>

// user include files

// Upgrade Board
#include "L1Trigger/L1TGlobal/interface/GtBoard.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

class GlobalStableParameters;
class L1GtParameters;
class L1GtBoardMaps;

class L1GtPrescaleFactors;
class L1GtTriggerMask;

class TriggerMenu;

// class declaration


  class L1TGlobalProducer : public edm::EDProducer
{

public:

    explicit L1TGlobalProducer(const edm::ParameterSet&);
    ~L1TGlobalProducer();

    virtual void produce(edm::Event&, const edm::EventSetup&);
    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
    // return pointer to uGt GtBoard  QUESTION: Is this used anywhere?
    //inline const l1t::GtBoard* gtBrd() const
    //{
    //    return m_uGtBrd;
    //}     

private:

    /// cached stuff

    /// stable parameters
    const GlobalStableParameters* m_l1GtStablePar;
    unsigned long long m_l1GtStableParCacheID;

    // trigger menu
    const TriggerMenu* m_l1GtMenu;
    unsigned long long m_l1GtMenuCacheID;

    /// number of physics triggers
    unsigned int m_numberPhysTriggers;

    /// number of DAQ partitions
    unsigned int m_numberDaqPartitions;

    /// number of objects of each type
    ///    { Mu, NoIsoEG, IsoEG, Jet, Tau, ETM, ETT, HTT, JetCounts };
    int m_nrL1Mu;
    int m_nrL1EG;
    int m_nrL1Tau;    

    int m_nrL1Jet;

//  *** ??? Do we still need this?
    int m_nrL1JetCounts;

    // ... the rest of the objects are global

    int m_ifMuEtaNumberBits;
    int m_ifCaloEtaNumberBits;


    /// parameters
    const L1GtParameters* m_l1GtPar;
    unsigned long long m_l1GtParCacheID;

    ///    total number of Bx's in the event coming from EventSetup
    int m_totalBxInEvent;

    ///    active boards in L1 GT DAQ record 
    boost::uint16_t m_activeBoardsGtDaq;

    /// length of BST record (in bytes) from event setup
    unsigned int m_bstLengthBytes;

    /// board maps - cache only the record
    const L1GtBoardMaps* m_l1GtBM;
    unsigned long long m_l1GtBMCacheID;


    /// prescale factors
    const L1GtPrescaleFactors* m_l1GtPfAlgo;
    unsigned long long m_l1GtPfAlgoCacheID;



    const std::vector<std::vector<int> >* m_prescaleFactorsAlgoTrig;
    std::vector<std::vector<int> > m_initialPrescaleFactorsAlgoTrig;

    /// CSV file for prescales
    std::string m_prescalesFile;


    /// trigger masks & veto masks
    const L1GtTriggerMask* m_l1GtTmAlgo;
    unsigned long long m_l1GtTmAlgoCacheID;

    const L1GtTriggerMask* m_l1GtTmVetoAlgo;
    unsigned long long m_l1GtTmVetoAlgoCacheID;


    const std::vector<unsigned int>* m_triggerMaskAlgoTrig;
    std::vector<unsigned int> m_initialTriggerMaskAlgoTrig;

    const std::vector<unsigned int>* m_triggerMaskVetoAlgoTrig;
    std::vector<unsigned int> m_initialTriggerMaskVetoAlgoTrig;

private:

/*
    GtProducerPSB* m_gtPSB;
    GtProducerGTL* m_gtGTL;
    GtProducerFDL* m_gtFDL;
*/
    l1t::GtBoard* m_uGtBrd;

    /// input tag for muon collection from GMT
    edm::InputTag m_muInputTag;
    edm::EDGetTokenT<BXVector<l1t::Muon>> m_muInputToken;

    /// input tag for calorimeter collections from GCT
    edm::InputTag m_egInputTag;
    edm::InputTag m_tauInputTag;
    edm::InputTag m_jetInputTag;
    edm::InputTag m_sumInputTag;
    edm::EDGetTokenT<BXVector<l1t::EGamma>> m_egInputToken;
    edm::EDGetTokenT<BXVector<l1t::Tau>> m_tauInputToken;
    edm::EDGetTokenT<BXVector<l1t::Jet>> m_jetInputToken;
    edm::EDGetTokenT<BXVector<l1t::EtSum>> m_sumInputToken;

    /// input tag for external conditions
    edm::InputTag m_extInputTag;
    edm::EDGetTokenT<BXVector<GlobalExtBlk>> m_extInputToken;

    /// logical flag to produce the L1 GT DAQ readout record
    bool m_produceL1GtDaqRecord;

    /// logical flag to produce the L1 GT object map record
    bool m_produceL1GtObjectMapRecord;

    /// logical flag to write the PSB content in the  L1 GT DAQ record
    bool m_writePsbL1GtDaqRecord;

    /// number of "bunch crossing in the event" (BxInEvent) to be emulated
    /// symmetric around L1Accept (BxInEvent = 0):
    ///    1 (BxInEvent = 0); 3 (F 0 1) (standard record); 5 (E F 0 1 2) (debug record)
    /// even numbers (except 0) "rounded" to the nearest lower odd number
    int m_emulateBxInEvent;

    /// Bx expected in Data coming to GT
    int m_L1DataBxInEvent;

    /// alternative for number of BX per active board in GT DAQ record: 0 or 1
    /// the position is identical with the active board bit
    unsigned int m_alternativeNrBxBoardDaq;

    /// length of BST record (in bytes) from parameter set
    int m_psBstLengthBytes;


    /// prescale set used
    unsigned int m_prescaleSet;

    /// run algorithm triggers
    ///     if true, unprescaled (all prescale factors 1)
    ///     will overwrite the event setup
    bool m_algorithmTriggersUnprescaled;

    ///     if true, unmasked - all enabled (all trigger masks set to 0)
    ///     will overwrite the event setup
    bool m_algorithmTriggersUnmasked;


private:

    /// verbosity level
    int m_verbosity;
    bool m_isDebugEnabled;

};


#endif /*GtProducer_h*/

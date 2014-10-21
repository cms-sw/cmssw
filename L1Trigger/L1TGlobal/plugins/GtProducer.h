#ifndef GtProducer_h
#define GtProducer_h

/**
 * \class GtProducer
 *
 *
 * Description: L1 Global Trigger producer.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Brian Winer  Ohio State
 *
 * $Date$
 * $Revision$
 *
 * The CMSSW implementation based on Legacy System Code
 *
 */

// system include files
#include <string>
#include <vector>

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

// class declaration

namespace l1t {

class GtProducer : public edm::EDProducer
{

public:

    explicit GtProducer(const edm::ParameterSet&);
    ~GtProducer();

    virtual void produce(edm::Event&, const edm::EventSetup&);

    // return pointer to uGt GtBoard  QUESTION: Is this used anywhere?
    inline const GtBoard* gtBrd() const
    {
        return m_uGtBrd;
    }    

private:

    /// cached stuff

    /// stable parameters
    const GlobalStableParameters* m_l1GtStablePar;
    unsigned long long m_l1GtStableParCacheID;

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


    /// trigger masks & veto masks
    const L1GtTriggerMask* m_l1GtTmAlgo;
    unsigned long long m_l1GtTmAlgoCacheID;

    const L1GtTriggerMask* m_l1GtTmVetoAlgo;
    unsigned long long m_l1GtTmVetoAlgoCacheID;


    std::vector<unsigned int> m_triggerMaskAlgoTrig;

    std::vector<unsigned int> m_triggerMaskVetoAlgoTrig;

private:

/*
    GtProducerPSB* m_gtPSB;
    GtProducerGTL* m_gtGTL;
    GtProducerFDL* m_gtFDL;
*/
    GtBoard* m_uGtBrd;

    /// input tag for muon collection from GMT
    edm::InputTag m_muInputTag;

    /// input tag for calorimeter collections from GCT
    edm::InputTag m_caloInputTag;

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

}
#endif /*GtProducer_h*/

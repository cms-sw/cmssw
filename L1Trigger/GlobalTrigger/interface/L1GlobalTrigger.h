#ifndef GlobalTrigger_L1GlobalTrigger_h
#define GlobalTrigger_L1GlobalTrigger_h

/**
 * \class L1GlobalTrigger
 * 
 * 
 * Description: L1 Global Trigger producer.
 *  
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 * $Date$
 * $Revision$
 *
 * The CMSSW implementation of the L1 Global Trigger emulator
 * uses concepts and code from the ORCA L1 Global Trigger simulation,
 * authors: N. Neumeister, M. Fierro, M. Eder  - HEPHY Vienna.
 *  
 */

// system include files
#include <string>
#include <vector>

#include <boost/cstdint.hpp>

// user include files

#include "CondFormats/L1TObjects/interface/L1GtFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtBoard.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

// forward classes
class L1GlobalTriggerPSB;
class L1GlobalTriggerGTL;
class L1GlobalTriggerFDL;

class L1GtStableParameters;
class L1GtParameters;
class L1GtBoardMaps;

class L1GtPrescaleFactors;
class L1GtTriggerMask;

// class declaration
class L1GlobalTrigger : public edm::EDProducer
{

public:

    explicit L1GlobalTrigger(const edm::ParameterSet&);
    ~L1GlobalTrigger();

    virtual void produce(edm::Event&, const edm::EventSetup&);

    // return pointer to PSB
    inline const L1GlobalTriggerPSB* gtPSB() const
    {
        return m_gtPSB;
    }

    // return pointer to GTL
    inline const L1GlobalTriggerGTL* gtGTL() const
    {
        return m_gtGTL;
    }

    // return pointer to FDL
    inline const L1GlobalTriggerFDL* gtFDL() const
    {
        return m_gtFDL;
    }

private:
    
    /// cached stuff

    /// stable parameters
    const L1GtStableParameters* m_l1GtStablePar;
    unsigned long long m_l1GtStableParCacheID;

    /// number of physics triggers
    unsigned int m_numberPhysTriggers;
    
    /// number of objects of each type
    ///    { Mu, NoIsoEG, IsoEG, CenJet, ForJet, TauJet, ETM, ETT, HTT, JetCounts };
    unsigned int m_nrL1Mu;

    unsigned int m_nrL1NoIsoEG;
    unsigned int m_nrL1IsoEG;

    unsigned int m_nrL1CenJet;
    unsigned int m_nrL1ForJet;
    unsigned int m_nrL1TauJet;

    // ... the rest of the objects are global
    
    

    /// parameters
    const L1GtParameters* m_l1GtPar;
    unsigned long long m_l1GtParCacheID;
    
    ///    total number of Bx's in the event
    int m_totalBxInEvent;

    ///    active boards in L1 GT DAQ record and in L1 GT EVM record
    boost::uint16_t m_activeBoardsGtDaq;
    boost::uint16_t m_activeBoardsGtEvm;

    /// board maps - cache only the record
    const L1GtBoardMaps* m_l1GtBM;
    unsigned long long m_l1GtBMCacheID;
    
    
    ///prescale factors & trigger masks
    const L1GtPrescaleFactors* m_l1GtPf;
    unsigned long long m_l1GtPfCacheID;
    
    std::vector<int> m_prescaleFactors;

    const L1GtTriggerMask* m_l1GtTm;
    unsigned long long m_l1GtTmCacheID;
 
    std::vector<unsigned int> m_triggerMask;

private:


    L1GlobalTriggerPSB* m_gtPSB;
    L1GlobalTriggerGTL* m_gtGTL;
    L1GlobalTriggerFDL* m_gtFDL;

    /// input tag for muon collection from GMT
    edm::InputTag m_muGmtInputTag;

    /// input tag for calorimeter collections from GCT
    edm::InputTag m_caloGctInputTag;


    /// logical flag to produce the L1 GT DAQ readout record
    bool m_produceL1GtDaqRecord;

    /// logical flag to produce the L1 GT EVM readout record
    bool m_produceL1GtEvmRecord;

    /// logical flag to produce the L1 GT object map record
    bool m_produceL1GtObjectMapRecord;

    // logical flag to write the PSB content in the  L1 GT DAQ record
    bool m_writePsbL1GtDaqRecord;

};

#endif /*GlobalTrigger_L1GlobalTrigger_h*/

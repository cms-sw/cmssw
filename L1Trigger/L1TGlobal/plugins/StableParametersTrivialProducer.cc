/**
 * \class StableParametersTrivialProducer
 * 
 * 
 * Description: ESProducer for L1 GT parameters.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 *
 */

// this class header
#include "StableParametersTrivialProducer.h"

// system include files
#include <memory>
#include <vector>

#include "boost/shared_ptr.hpp"

// user include files
//   base class
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include "CondFormats/DataRecord/interface/L1TGlobalStableParametersRcd.h"

// forward declarations

// constructor(s)
StableParametersTrivialProducer::StableParametersTrivialProducer(
    const edm::ParameterSet& parSet) {

    // tell the framework what data is being produced
    setWhatProduced(
        this, &StableParametersTrivialProducer::produceGtStableParameters);

    // now do what ever other initialization is needed

    // trigger decision

    // number of physics trigger algorithms
    m_numberPhysTriggers
        = parSet.getParameter<unsigned int>("NumberPhysTriggers");

    // additional number of physics trigger algorithms
    m_numberPhysTriggersExtended
        = parSet.getParameter<unsigned int>("NumberPhysTriggersExtended");

    // number of technical triggers
    m_numberTechnicalTriggers
        = parSet.getParameter<unsigned int>("NumberTechnicalTriggers");

    // trigger objects

    // muons
    m_numberL1Mu = parSet.getParameter<unsigned int>("NumberL1Mu");

    // e/gamma and isolated e/gamma objects
    m_numberL1NoIsoEG = parSet.getParameter<unsigned int>("NumberL1NoIsoEG");
    m_numberL1IsoEG = parSet.getParameter<unsigned int>("NumberL1IsoEG");

    // central, forward and tau jets
    m_numberL1CenJet = parSet.getParameter<unsigned int>("NumberL1CenJet");
    m_numberL1ForJet = parSet.getParameter<unsigned int>("NumberL1ForJet");
    m_numberL1TauJet = parSet.getParameter<unsigned int>("NumberL1TauJet");

    // jet counts
    m_numberL1JetCounts = parSet.getParameter<unsigned int>("NumberL1JetCounts");

    // hardware

    // number of maximum chips defined in the xml file
    m_numberConditionChips
        = parSet.getParameter<unsigned int>("NumberConditionChips");

    // number of pins on the GTL condition chips
    m_pinsOnConditionChip
        = parSet.getParameter<unsigned int>("PinsOnConditionChip");

    // correspondence "condition chip - GTL algorithm word" in the hardware
    // e.g.: chip 2: 0 - 95;  chip 1: 96 - 128 (191)
    m_orderConditionChip
        = parSet.getParameter<std::vector<int> >("OrderConditionChip");

    // number of PSB boards in GT
    m_numberPsbBoards = parSet.getParameter<int>("NumberPsbBoards");

    /// number of bits for eta of calorimeter objects
    m_ifCaloEtaNumberBits
        = parSet.getParameter<unsigned int>("IfCaloEtaNumberBits");

    /// number of bits for eta of calorimeter objects
    m_ifMuEtaNumberBits = parSet.getParameter<unsigned int>("IfMuEtaNumberBits");

    // GT DAQ record organized in words of WordLength bits
    m_wordLength = parSet.getParameter<int>("WordLength");

    // one unit in the word is UnitLength bits
    m_unitLength = parSet.getParameter<int>("UnitLength");

}

// destructor
StableParametersTrivialProducer::~StableParametersTrivialProducer() {

    // empty

}

// member functions

// method called to produce the data
boost::shared_ptr<GlobalStableParameters> 
    StableParametersTrivialProducer::produceGtStableParameters(
        const L1TGlobalStableParametersRcd& iRecord) {

    boost::shared_ptr<GlobalStableParameters> pL1uGtStableParameters =
        boost::shared_ptr<GlobalStableParameters>(new GlobalStableParameters());

    // set the number of physics trigger algorithms
    pL1uGtStableParameters->setGtNumberPhysTriggers(m_numberPhysTriggers);

    // set the additional number of physics trigger algorithms
    pL1uGtStableParameters->setGtNumberPhysTriggersExtended(m_numberPhysTriggersExtended);

    // set the number of technical triggers
    pL1uGtStableParameters->setGtNumberTechnicalTriggers(m_numberTechnicalTriggers);

    // set the number of L1 muons received by GT
    pL1uGtStableParameters->setGtNumberL1Mu(m_numberL1Mu);
    
    //  set the number of L1 e/gamma objects received by GT
    pL1uGtStableParameters->setGtNumberL1NoIsoEG(m_numberL1NoIsoEG);
    
    //  set the number of L1 isolated e/gamma objects received by GT
    pL1uGtStableParameters->setGtNumberL1IsoEG(m_numberL1IsoEG);
    
    // set the number of L1 central jets received by GT
    pL1uGtStableParameters->setGtNumberL1CenJet(m_numberL1CenJet);
    
    // set the number of L1 forward jets received by GT
    pL1uGtStableParameters->setGtNumberL1ForJet(m_numberL1ForJet);
    
    // set the number of L1 tau jets received by GT
    pL1uGtStableParameters->setGtNumberL1TauJet(m_numberL1TauJet);
    
    // set the number of L1 jet counts received by GT
    pL1uGtStableParameters->setGtNumberL1JetCounts(m_numberL1JetCounts);
    
    // hardware stuff
    
    // set the number of condition chips in GTL
    pL1uGtStableParameters->setGtNumberConditionChips(m_numberConditionChips);
    
    // set the number of pins on the GTL condition chips
    pL1uGtStableParameters->setGtPinsOnConditionChip(m_pinsOnConditionChip);
    
    // set the correspondence "condition chip - GTL algorithm word"
    // in the hardware
    pL1uGtStableParameters->setGtOrderConditionChip(m_orderConditionChip);
    
    // set the number of PSB boards in GT
    pL1uGtStableParameters->setGtNumberPsbBoards(m_numberPsbBoards);
    
    //   set the number of bits for eta of calorimeter objects
    pL1uGtStableParameters->setGtIfCaloEtaNumberBits(m_ifCaloEtaNumberBits);
    
    //   set the number of bits for eta of muon objects
    pL1uGtStableParameters->setGtIfMuEtaNumberBits(m_ifMuEtaNumberBits);
    
    // set WordLength
    pL1uGtStableParameters->setGtWordLength(m_wordLength);
    
    // set one UnitLength
    pL1uGtStableParameters->setGtUnitLength(m_unitLength);
    
    //
    //
    return pL1uGtStableParameters;

}

DEFINE_FWK_EVENTSETUP_MODULE(StableParametersTrivialProducer);

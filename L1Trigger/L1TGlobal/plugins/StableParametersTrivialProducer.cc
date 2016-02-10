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

#include "CondFormats/DataRecord/interface/L1TGlobalParametersRcd.h"

// forward declarations

// constructor(s)
StableParametersTrivialProducer::StableParametersTrivialProducer(
    const edm::ParameterSet& parSet) {

    // tell the framework what data is being produced
    setWhatProduced(
        this, &StableParametersTrivialProducer::produceGtStableParameters);

    // now do what ever other initialization is needed


    // bx in event
    m_totalBxInEvent = parSet.getParameter<int>("NumberBxInEvent");
  
    // trigger decision

    // number of physics trigger algorithms
    m_numberPhysTriggers
        = parSet.getParameter<unsigned int>("NumberPhysTriggers");

    // trigger objects

    // muons
    m_numberL1Mu = parSet.getParameter<unsigned int>("NumberL1Mu");

    // e/gamma and isolated e/gamma objects
    m_numberL1EG = parSet.getParameter<unsigned int>("NumberL1EG");

    //  jets
    m_numberL1Jet = parSet.getParameter<unsigned int>("NumberL1Jet");
    
    //  tau
    m_numberL1Tau = parSet.getParameter<unsigned int>("NumberL1Tau");


    // hardware

    // number of maximum chips defined in the xml file
    m_numberChips
        = parSet.getParameter<unsigned int>("NumberChips");

    // number of pins on the GTL condition chips
    m_pinsOnChip
        = parSet.getParameter<unsigned int>("PinsOnChip");

    // correspondence "condition chip - GTL algorithm word" in the hardware
    // e.g.: chip 2: 0 - 95;  chip 1: 96 - 128 (191)
    m_orderOfChip
        = parSet.getParameter<std::vector<int> >("OrderOfChip");

/*
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
*/
}

// destructor
StableParametersTrivialProducer::~StableParametersTrivialProducer() {

    // empty

}

// member functions

// method called to produce the data
boost::shared_ptr<L1TGlobalParameters> 
    StableParametersTrivialProducer::produceGtStableParameters(
        const L1TGlobalParametersRcd& iRecord) {

    boost::shared_ptr<L1TGlobalParameters> pL1uGtStableParameters =
        boost::shared_ptr<L1TGlobalParameters>(new L1TGlobalParameters());

    // set the number of bx in event
    pL1uGtStableParameters->setGtTotalBxInEvent(m_totalBxInEvent);

    // set the number of physics trigger algorithms
    pL1uGtStableParameters->setGtNumberPhysTriggers(m_numberPhysTriggers);

    // set the number of L1 muons received by GT
    pL1uGtStableParameters->setGtNumberL1Mu(m_numberL1Mu);
    
    //  set the number of L1 e/gamma objects received by GT
    pL1uGtStableParameters->setGtNumberL1EG(m_numberL1EG);
       
    // set the number of L1 central jets received by GT
    pL1uGtStableParameters->setGtNumberL1Jet(m_numberL1Jet);
        
    // set the number of L1 tau jets received by GT
    pL1uGtStableParameters->setGtNumberL1Tau(m_numberL1Tau);
    
   
    // hardware stuff
    
    // set the number of condition chips in GTL
    pL1uGtStableParameters->setGtNumberChips(m_numberChips);
    
    // set the number of pins on the GTL condition chips
    pL1uGtStableParameters->setGtPinsOnChip(m_pinsOnChip);
    
    // set the correspondence "condition chip - GTL algorithm word"
    // in the hardware
    pL1uGtStableParameters->setGtOrderOfChip(m_orderOfChip);
/*    
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
*/    
    //
    //
    return pL1uGtStableParameters;

}

DEFINE_FWK_EVENTSETUP_MODULE(StableParametersTrivialProducer);

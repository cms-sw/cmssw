#ifndef L1Trigger_L1TGlobal_StableParametersTrivialProducer_h
#define L1Trigger_L1TGlobal_StableParametersTrivialProducer_h

/**
 * \class StableParametersTrivialProducer
 * 
 * 
 * Description: ESProducer for L1 GT parameters.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 *
 */

// system include files
#include <memory>
#include <iostream>
#include <vector>

#include "boost/shared_ptr.hpp"
#include <boost/cstdint.hpp>

// user include files
//   base class
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1TGlobalParameters.h"
#include "CondFormats/DataRecord/interface/L1TGlobalParametersRcd.h"

// forward declarations

// class declaration
class StableParametersTrivialProducer : public edm::ESProducer
{

public:

    /// constructor
    StableParametersTrivialProducer(const edm::ParameterSet&);

    /// destructor
    ~StableParametersTrivialProducer();

    /// public methods

    /// L1 GT parameters
    boost::shared_ptr<L1TGlobalParameters> produceGtStableParameters(
        const L1TGlobalParametersRcd&);

private:


    /// bx in event
    int m_totalBxInEvent; 

    /// trigger decision

    /// number of physics trigger algorithms
    unsigned int m_numberPhysTriggers;


    /// trigger objects

    /// muons
    unsigned int m_numberL1Mu;

    /// e/gamma and isolated e/gamma objects
    unsigned int m_numberL1EG;


    /// jets
    unsigned int m_numberL1Jet;
    
    /// tau
    unsigned int m_numberL1Tau;


private:

    /// hardware

    /// number of maximum chips defined in the xml file
    unsigned int m_numberChips;

    /// number of pins on the GTL condition chips
    unsigned int m_pinsOnChip;

    /// correspondence "condition chip - GTL algorithm word" in the hardware
    /// e.g.: chip 2: 0 - 95;  chip 1: 96 - 128 (191)
    std::vector<int> m_orderOfChip;
/*
    /// number of PSB boards in GT
    int m_numberPsbBoards;

    /// number of bits for eta of calorimeter objects
    unsigned int m_ifCaloEtaNumberBits;

    /// number of bits for eta of muon objects
    unsigned int m_ifMuEtaNumberBits;

private:

    /// GT DAQ record organized in words of WordLength bits
    int m_wordLength;

    /// one unit in the word is UnitLength bits
    int m_unitLength;
*/
};

#endif

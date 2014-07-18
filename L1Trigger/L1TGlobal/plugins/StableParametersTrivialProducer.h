#ifndef L1GtConfigProducers_StableParametersTrivialProducer_h
#define L1GtConfigProducers_StableParametersTrivialProducer_h

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

#include <vector>

#include "boost/shared_ptr.hpp"
#include <boost/cstdint.hpp>

// user include files
//   base class
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/GlobalStableParameters.h"
#include "CondFormats/DataRecord/interface/L1TGlobalStableParametersRcd.h"

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
    boost::shared_ptr<GlobalStableParameters> produceGtStableParameters(
        const L1TGlobalStableParametersRcd&);

private:

    /// trigger decision

    /// number of physics trigger algorithms
    unsigned int m_numberPhysTriggers;

    /// additional number of physics trigger algorithms
    unsigned int m_numberPhysTriggersExtended;

    /// number of technical triggers
    unsigned int m_numberTechnicalTriggers;

    /// trigger objects

    /// muons
    unsigned int m_numberL1Mu;

    /// e/gamma and isolated e/gamma objects
    unsigned int m_numberL1NoIsoEG;
    unsigned int m_numberL1IsoEG;

    /// central, forward and tau jets
    unsigned int m_numberL1CenJet;
    unsigned int m_numberL1ForJet;
    unsigned int m_numberL1TauJet;

    /// jet counts
    unsigned int m_numberL1JetCounts;

private:

    /// hardware

    /// number of maximum chips defined in the xml file
    unsigned int m_numberConditionChips;

    /// number of pins on the GTL condition chips
    unsigned int m_pinsOnConditionChip;

    /// correspondence "condition chip - GTL algorithm word" in the hardware
    /// e.g.: chip 2: 0 - 95;  chip 1: 96 - 128 (191)
    std::vector<int> m_orderConditionChip;

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

};

#endif

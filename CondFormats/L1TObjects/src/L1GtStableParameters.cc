/**
 * \class L1GtStableParameters
 * 
 * 
 * Description: L1 GT stable parameters.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "CondFormats/L1TObjects/interface/L1GtStableParameters.h"

// system include files
#include <vector>

#include <ostream>
#include <iomanip>

#include <boost/cstdint.hpp>

// user include files
//   base class

// forward declarations

// constructor
L1GtStableParameters::L1GtStableParameters()
{
    // empty
}

// destructor
L1GtStableParameters::~L1GtStableParameters()
{
    // empty
}

// set the number of physics trigger algorithms
void L1GtStableParameters::setGtNumberPhysTriggers(
    const unsigned int& numberPhysTriggersValue)
{

    m_numberPhysTriggers = numberPhysTriggersValue;

}

// set the additional number of physics trigger algorithms
void L1GtStableParameters::setGtNumberPhysTriggersExtended(
    const unsigned int& numberPhysTriggersExtendedValue)
{

    m_numberPhysTriggersExtended = numberPhysTriggersExtendedValue;

}

// set the number of technical triggers
void L1GtStableParameters::setGtNumberTechnicalTriggers(
    const unsigned int& numberTechnicalTriggersValue)
{

    m_numberTechnicalTriggers = numberTechnicalTriggersValue;

}


// set the number of L1 muons received by GT
void L1GtStableParameters::setGtNumberL1Muons(const unsigned int& numberL1MuonsValue)
{

    m_numberL1Muons = numberL1MuonsValue;

}


//  set the number of L1 e/gamma objects received by GT
void L1GtStableParameters::setGtNumberL1EGamma(const unsigned int& numberL1EGammaValue)
{

    m_numberL1EGamma = numberL1EGammaValue;

}

//  set the number of L1 isolated e/gamma objects received by GT
void L1GtStableParameters::setGtNumberL1IsolatedEGamma(
    const unsigned int& numberL1IsolatedEGammaValue)
{

    m_numberL1IsolatedEGamma = numberL1IsolatedEGammaValue;

}

// set the number of L1 central jets received by GT
void L1GtStableParameters::setGtNumberL1CentralJets(
    const unsigned int& numberL1CentralJetsValue)
{

    m_numberL1CentralJets = numberL1CentralJetsValue;

}

// set the number of L1 forward jets received by GT
void L1GtStableParameters::setGtNumberL1ForwardJets(
    const unsigned int& numberL1ForwardJetsValue)
{

    m_numberL1ForwardJets = numberL1ForwardJetsValue;

}


// set the number of L1 tau jets received by GT
void L1GtStableParameters::setGtNumberL1TauJets(
    const unsigned int& numberL1TauJetsValue)
{

    m_numberL1TauJets = numberL1TauJetsValue;

}

// set the number of L1 jet counts received by GT
void L1GtStableParameters::setGtNumberL1JetCounts(
    const unsigned int& numberL1JetCountsValue)
{

    m_numberL1JetCounts = numberL1JetCountsValue;

}

// hardware stuff

// set the number of condition chips in GTL
void L1GtStableParameters::setGtNumberConditionChips(
    const unsigned int& numberConditionChipsValue)
{

    m_numberConditionChips = numberConditionChipsValue;

}


// set the number of pins on the GTL condition chips
void L1GtStableParameters::setGtPinsOnConditionChip(
    const unsigned int& pinsOnConditionChipValue)
{

    m_pinsOnConditionChip = pinsOnConditionChipValue;

}



// set the correspondence "condition chip - GTL algorithm word"
// in the hardware
void L1GtStableParameters::setGtOrderConditionChip(
    const std::vector<int>& orderConditionChipValue)
{

    m_orderConditionChip = orderConditionChipValue;

}


// set the number of PSB boards in GT
void L1GtStableParameters::setGtNumberPsbBoards(const int& numberPsbBoardsValue)
{

    m_numberPsbBoards = numberPsbBoardsValue;

}

// set WordLength
void L1GtStableParameters::setGtWordLength(const int& wordLengthValue)
{

    m_wordLength = wordLengthValue;

}


// set one UnitLength
void L1GtStableParameters::setGtUnitLength(const int& unitLengthValue)
{

    m_unitLength = unitLengthValue;

}


// print all the L1 GT stable parameters
void L1GtStableParameters::print(std::ostream& myStr) const
{
    myStr << "\nL1 GT Stable Parameters \n" << std::endl;


    /// trigger decision

    /// number of physics trigger algorithms
    myStr << "\n  Number of physics trigger algorithms =            "
    << m_numberPhysTriggers << std::endl;

    /// additional number of physics trigger algorithms
    myStr << "  Additional number of physics trigger algorithms = "
    << m_numberPhysTriggersExtended << std::endl;

    /// number of technical triggers
    myStr << "  Number of technical triggers =                    "
    << m_numberTechnicalTriggers << std::endl;


    /// muons
    myStr << "\n  Number of muon received by L1 GT =                     "
    << m_numberL1Muons << std::endl;

    /// e/gamma and isolated e/gamma objects
    myStr << "  Number of e/gamma objects received by L1 GT =          "
    << m_numberL1EGamma << std::endl;
    myStr << "  Number of isolated e/gamma objects received by L1 GT = "
    <<  m_numberL1IsolatedEGamma << std::endl;

    /// central, forward and tau jets
    myStr << "\n  Number of central jets received by L1 GT =             "
    << m_numberL1CentralJets << std::endl;
    myStr << "  Number of forward jets received by L1 GT =             "
    << m_numberL1ForwardJets << std::endl;
    myStr << "  Number of tau jets received by L1 GT =                 "
    << m_numberL1TauJets << std::endl;

    /// jet counts
    myStr << "\n  Number of jet counts received by L1 GT =               "
    << m_numberL1JetCounts << std::endl;


    /// hardware

    /// number of condition chips
    myStr << "\n  Number of condition chips =                        "
    << m_numberConditionChips << std::endl;

    /// number of pins on the GTL condition chips
    myStr << "  Number of pins on the GTL condition chips =        "
    << m_pinsOnConditionChip << std::endl;

    /// correspondence "condition chip - GTL algorithm word" in the hardware
    /// chip 2: 0 - 95;  chip 1: 96 - 128 (191)
    myStr << "  Order of condition chips for GTL algorithm word = {";

    for (unsigned int iChip = 0; iChip < m_orderConditionChip.size(); ++iChip) {
        myStr << m_orderConditionChip[iChip];
        if (iChip != (m_orderConditionChip.size() - 1)) {
            myStr << ", ";
        }
    }

    myStr << "}" << std::endl;

    /// number of PSB boards in GT
    myStr << "\n  Number of PSB boards in GT = " << m_numberPsbBoards << std::endl;


    /// GT DAQ record organized in words of WordLength bits
    myStr << "\n  Word length (bits) for GT records = " << m_wordLength << std::endl;

    /// one unit in the word is UnitLength bits
    myStr << "  Unit length (bits) for GT records = " << m_unitLength << std::endl;

    myStr << "\n" << std::endl;

}

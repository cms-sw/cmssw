/**
 * \class L1uGtStableParameters
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
#include "CondFormats/L1TObjects/interface/L1uGtStableParameters.h"

// system include files

#include <iomanip>


// user include files
//   base class

// forward declarations

// constructor
L1uGtStableParameters::L1uGtStableParameters() {
    // empty
}

// destructor
L1uGtStableParameters::~L1uGtStableParameters() {
    // empty
}

// set the number of physics trigger algorithms
void L1uGtStableParameters::setGtNumberPhysTriggers(
    const unsigned int& numberPhysTriggersValue) {

    m_numberPhysTriggers = numberPhysTriggersValue;

}

// set the additional number of physics trigger algorithms
void L1uGtStableParameters::setGtNumberPhysTriggersExtended(
    const unsigned int& numberPhysTriggersExtendedValue) {

    m_numberPhysTriggersExtended = numberPhysTriggersExtendedValue;

}

// set the number of technical triggers
void L1uGtStableParameters::setGtNumberTechnicalTriggers(
    const unsigned int& numberTechnicalTriggersValue) {

    m_numberTechnicalTriggers = numberTechnicalTriggersValue;

}

// set the number of L1 muons received by GT
void L1uGtStableParameters::setGtNumberL1Mu(const unsigned int& numberL1MuValue) {

    m_numberL1Mu = numberL1MuValue;

}

//  set the number of L1 e/gamma objects received by GT
void L1uGtStableParameters::setGtNumberL1NoIsoEG(
    const unsigned int& numberL1NoIsoEGValue) {

    m_numberL1NoIsoEG = numberL1NoIsoEGValue;

}

//  set the number of L1 isolated e/gamma objects received by GT
void L1uGtStableParameters::setGtNumberL1IsoEG(
    const unsigned int& numberL1IsoEGValue) {

    m_numberL1IsoEG = numberL1IsoEGValue;

}

// set the number of L1 central jets received by GT
void L1uGtStableParameters::setGtNumberL1CenJet(
    const unsigned int& numberL1CenJetValue) {

    m_numberL1CenJet = numberL1CenJetValue;

}

// set the number of L1 forward jets received by GT
void L1uGtStableParameters::setGtNumberL1ForJet(
    const unsigned int& numberL1ForJetValue) {

    m_numberL1ForJet = numberL1ForJetValue;

}

// set the number of L1 tau jets received by GT
void L1uGtStableParameters::setGtNumberL1TauJet(
    const unsigned int& numberL1TauJetValue) {

    m_numberL1TauJet = numberL1TauJetValue;

}

// set the number of L1 jet counts received by GT
void L1uGtStableParameters::setGtNumberL1JetCounts(
    const unsigned int& numberL1JetCountsValue) {

    m_numberL1JetCounts = numberL1JetCountsValue;

}

// hardware stuff

// set the number of condition chips in GTL
void L1uGtStableParameters::setGtNumberConditionChips(
    const unsigned int& numberConditionChipsValue) {

    m_numberConditionChips = numberConditionChipsValue;

}

// set the number of pins on the GTL condition chips
void L1uGtStableParameters::setGtPinsOnConditionChip(
    const unsigned int& pinsOnConditionChipValue) {

    m_pinsOnConditionChip = pinsOnConditionChipValue;

}

// set the correspondence "condition chip - GTL algorithm word"
// in the hardware
void L1uGtStableParameters::setGtOrderConditionChip(
    const std::vector<int>& orderConditionChipValue) {

    m_orderConditionChip = orderConditionChipValue;

}

// set the number of PSB boards in GT
void L1uGtStableParameters::setGtNumberPsbBoards(const int& numberPsbBoardsValue) {

    m_numberPsbBoards = numberPsbBoardsValue;

}

//   set the number of bits for eta of calorimeter objects
void L1uGtStableParameters::setGtIfCaloEtaNumberBits(
    const unsigned int& ifCaloEtaNumberBitsValue) {

    m_ifCaloEtaNumberBits = ifCaloEtaNumberBitsValue;

}

//   set the number of bits for eta of muon objects
void L1uGtStableParameters::setGtIfMuEtaNumberBits(
    const unsigned int& ifMuEtaNumberBitsValue) {

    m_ifMuEtaNumberBits = ifMuEtaNumberBitsValue;

}

// set WordLength
void L1uGtStableParameters::setGtWordLength(const int& wordLengthValue) {

    m_wordLength = wordLengthValue;

}

// set one UnitLength
void L1uGtStableParameters::setGtUnitLength(const int& unitLengthValue) {

    m_unitLength = unitLengthValue;

}

// print all the L1 GT stable parameters
void L1uGtStableParameters::print(std::ostream& myStr) const {
    myStr << "\nL1 GT Stable Parameters \n" << std::endl;

    // trigger decision

    // number of physics trigger algorithms
    myStr << "\n  Number of physics trigger algorithms =            "
        << m_numberPhysTriggers << std::endl;

    // additional number of physics trigger algorithms
    myStr << "  Additional number of physics trigger algorithms = "
        << m_numberPhysTriggersExtended << std::endl;

    // number of technical triggers
    myStr << "  Number of technical triggers =                    "
        << m_numberTechnicalTriggers << std::endl;

    // muons
    myStr << "\n  Number of muons received by L1 GT =                     "
        << m_numberL1Mu << std::endl;

    // e/gamma and isolated e/gamma objects
    myStr << "  Number of e/gamma objects received by L1 GT =          "
        << m_numberL1NoIsoEG << std::endl;
    myStr << "  Number of isolated e/gamma objects received by L1 GT = "
        << m_numberL1IsoEG << std::endl;

    // central, forward and tau jets
    myStr << "\n  Number of central jets received by L1 GT =             "
        << m_numberL1CenJet << std::endl;
    myStr << "  Number of forward jets received by L1 GT =             "
        << m_numberL1ForJet << std::endl;
    myStr << "  Number of tau jets received by L1 GT =                 "
        << m_numberL1TauJet << std::endl;

    // jet counts
    myStr << "\n  Number of jet counts received by L1 GT =               "
        << m_numberL1JetCounts << std::endl;

    // hardware

    // number of condition chips
    myStr << "\n  Number of condition chips =                        "
        << m_numberConditionChips << std::endl;

    // number of pins on the GTL condition chips
    myStr << "  Number of pins on the GTL condition chips =        "
        << m_pinsOnConditionChip << std::endl;

    // correspondence "condition chip - GTL algorithm word" in the hardware
    // chip 2: 0 - 95;  chip 1: 96 - 128 (191)
    myStr << "  Order of condition chips for GTL algorithm word = {";

    for (unsigned int iChip = 0; iChip < m_orderConditionChip.size(); ++iChip) {
        myStr << m_orderConditionChip[iChip];
        if (iChip != (m_orderConditionChip.size() - 1)) {
            myStr << ", ";
        }
    }

    myStr << "}" << std::endl;

    // number of PSB boards in GT
    myStr << "\n  Number of PSB boards in GT = " << m_numberPsbBoards
        << std::endl;

    // number of bits for eta of calorimeter objects
    myStr << "\n  Number of bits for eta of calorimeter objects = "
        << m_ifCaloEtaNumberBits << std::endl;

    // number of bits for eta of muon objects
    myStr << "\n  Number of bits for eta of muon objects = "
        << m_ifMuEtaNumberBits << std::endl;

    // GT DAQ record organized in words of WordLength bits
    myStr << "\n  Word length (bits) for GT records = " << m_wordLength
        << std::endl;

    // one unit in the word is UnitLength bits
    myStr << "  Unit length (bits) for GT records = " << m_unitLength
        << std::endl;

    myStr << "\n" << std::endl;

}

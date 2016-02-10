/**
 * \class L1TGlobalParameters
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
#include "CondFormats/L1TObjects/interface/L1TGlobalParameters.h"

// system include files

#include <iomanip>


// user include files
//   base class

// forward declarations

// constructor
L1TGlobalParameters::L1TGlobalParameters() {
    // empty
}

// destructor
L1TGlobalParameters::~L1TGlobalParameters() {
    // empty
}


// set the number of bx in event
void L1TGlobalParameters::setGtTotalBxInEvent(
    const int& numberBxValue) {

    m_totalBxInEvent = numberBxValue;

}



// set the number of physics trigger algorithms
void L1TGlobalParameters::setGtNumberPhysTriggers(
    const unsigned int& numberPhysTriggersValue) {

    m_numberPhysTriggers = numberPhysTriggersValue;

}


// set the number of L1 muons received by GT
void L1TGlobalParameters::setGtNumberL1Mu(const unsigned int& numberL1MuValue) {

    m_numberL1Mu = numberL1MuValue;

}

//  set the number of L1 e/gamma objects received by GT
void L1TGlobalParameters::setGtNumberL1EG(
    const unsigned int& numberL1EGValue) {

    m_numberL1EG = numberL1EGValue;

}


// set the number of L1 central jets received by GT
void L1TGlobalParameters::setGtNumberL1Jet(
    const unsigned int& numberL1JetValue) {

    m_numberL1Jet = numberL1JetValue;

}


// set the number of L1 tau jets received by GT
void L1TGlobalParameters::setGtNumberL1Tau(
    const unsigned int& numberL1TauValue) {

    m_numberL1Tau = numberL1TauValue;

}



// hardware stuff

// set the number of condition chips in GTL
void L1TGlobalParameters::setGtNumberChips(
    const unsigned int& numberChipsValue) {

    m_numberChips = numberChipsValue;

}

// set the number of pins on the GTL condition chips
void L1TGlobalParameters::setGtPinsOnChip(
    const unsigned int& pinsOnChipValue) {

    m_pinsOnChip = pinsOnChipValue;

}

// set the correspondence "condition chip - GTL algorithm word"
// in the hardware
void L1TGlobalParameters::setGtOrderOfChip(
    const std::vector<int>& orderOfChipValue) {

    m_orderOfChip = orderOfChipValue;

}
/*
// set the number of PSB boards in GT
void L1TGlobalParameters::setGtNumberPsbBoards(const int& numberPsbBoardsValue) {

    m_numberPsbBoards = numberPsbBoardsValue;

}

//   set the number of bits for eta of calorimeter objects
void L1TGlobalParameters::setGtIfCaloEtaNumberBits(
    const unsigned int& ifCaloEtaNumberBitsValue) {

    m_ifCaloEtaNumberBits = ifCaloEtaNumberBitsValue;

}

//   set the number of bits for eta of muon objects
void L1TGlobalParameters::setGtIfMuEtaNumberBits(
    const unsigned int& ifMuEtaNumberBitsValue) {

    m_ifMuEtaNumberBits = ifMuEtaNumberBitsValue;

}

// set WordLength
void L1TGlobalParameters::setGtWordLength(const int& wordLengthValue) {

    m_wordLength = wordLengthValue;

}

// set one UnitLength
void L1TGlobalParameters::setGtUnitLength(const int& unitLengthValue) {

    m_unitLength = unitLengthValue;

}
*/
// print all the L1 GT stable parameters
void L1TGlobalParameters::print(std::ostream& myStr) const {
    myStr << "\nL1T Global  Parameters \n" << std::endl;


    // number of bx
    myStr << "\n  Number of bx in Event =            "
        << m_totalBxInEvent << std::endl;

    // trigger decision

    // number of physics trigger algorithms
    myStr << "\n  Number of physics trigger algorithms =            "
        << m_numberPhysTriggers << std::endl;

    // muons
    myStr << "\n  Number of muons received by L1 GT =                     "
        << m_numberL1Mu << std::endl;

    // e/gamma and isolated e/gamma objects
    myStr << "  Number of e/gamma objects received by L1 GT =          "
        << m_numberL1EG << std::endl;

    // central, forward and tau jets
    myStr << "\n  Number of  jets received by L1 GT =             "
        << m_numberL1Jet << std::endl;

    myStr << "  Number of tau  received by L1 GT =                 "
        << m_numberL1Tau << std::endl;


    // hardware

    // number of condition chips
    myStr << "\n  Number of condition chips =                        "
        << m_numberChips << std::endl;

    // number of pins on the GTL condition chips
    myStr << "  Number of pins on chips =        "
        << m_pinsOnChip << std::endl;

    // correspondence "condition chip - GTL algorithm word" in the hardware
    // chip 2: 0 - 95;  chip 1: 96 - 128 (191)
    myStr << "  Order of  chips for algorithm word = {";

    for (unsigned int iChip = 0; iChip < m_orderOfChip.size(); ++iChip) {
        myStr << m_orderOfChip[iChip];
        if (iChip != (m_orderOfChip.size() - 1)) {
            myStr << ", ";
        }
    }

    myStr << "}" << std::endl;
/*
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
*/
    myStr << "\n" << std::endl;

}

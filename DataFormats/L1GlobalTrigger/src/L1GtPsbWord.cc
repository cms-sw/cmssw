/**
 * \class L1GtPsbWord
 * 
 * 
 * 
 * Description: see header file 
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
#include "DataFormats/L1GlobalTrigger/interface/L1GtPsbWord.h"

// system include files

// user include files
#include "FWCore/Utilities/interface/EDMException.h"


// constructors

// empty constructor, all members set to zero;
L1GtPsbWord::L1GtPsbWord() {

    m_boardId = 0;
    m_bxInEvent = 0;
    m_bxNr = 0;
    m_eventNr = 0;

    for (int iA = 0; iA < NumberAData; ++iA) {
        m_aData[iA] = 0;            
    }

    for (int iB = 0; iB < NumberBData; ++iB) {
        m_bData[iB] = 0;            
    }

    m_localBxNr = 0;

}

// constructor from unpacked values;
L1GtPsbWord::L1GtPsbWord(
    uint16_t boardIdValue,
    uint16_t bxInEventValue,
    uint16_t bxNrValue,
    uint32_t eventNrValue,
    uint16_t aDataValue[NumberAData],
    uint16_t bDataValue[NumberBData],
    uint16_t localBxNrValue
    ) 
{

    m_boardId = boardIdValue;
    m_bxInEvent = bxInEventValue;
    m_bxNr = bxNrValue;
    m_eventNr = eventNrValue;

    for (int iA = 0; iA < NumberAData; ++iA) {
        m_aData[iA] = aDataValue[iA];            
    }

    for (int iB = 0; iB < NumberBData; ++iB) {
        m_bData[iB] = bDataValue[iB];            
    }

    m_localBxNr = localBxNrValue;
    
}

// destructor
L1GtPsbWord::~L1GtPsbWord() {
}

// methods

// get/set A_DATA_CH_IA 

const uint16_t L1GtPsbWord::aData(int iA) const {

    if (iA < 0 || iA > NumberAData) {
        throw cms::Exception("aDataIndexError")
            << "\nError: index for A_DATA array out of range. Allowed range: [0, "
            << NumberAData << ") " << std::endl;
        
    } else {
         return m_aData[iA];         
    }     
     
}
void L1GtPsbWord::setAData(uint16_t aDataVal, int iA) { 

    if (iA < 0 || iA > NumberAData) {
        throw cms::Exception("aDataIndexError")
            << "\nError: index for A_DATA array out of range. Allowed range: [0, "
            << NumberAData << ") " << std::endl;
        
    } else {
        m_aData[iA] = aDataVal; 
    }     
    
}

// get/set B_DATA_CH_IB 

const uint16_t L1GtPsbWord::bData(int iB) const {

    if (iB < 0 || iB > NumberBData) {
        throw cms::Exception("bDataIndexError")
            << "\nError: index for B_DATA array out of range. Allowed range: [0, "
            << NumberBData << ") " << std::endl;
        
    } else {
         return m_bData[iB];         
    }     
     
}
void L1GtPsbWord::setBData(uint16_t bDataVal, int iB) { 

    if (iB < 0 || iB > NumberBData) {
        throw cms::Exception("bDataIndexError")
            << "\nError: index for B_DATA array out of range. Allowed range: [0, "
            << NumberBData << ") " << std::endl;
        
    } else {
        m_bData[iB] = bDataVal; 
    }     
    
}

// static class members
const int L1GtPsbWord::NumberAData;
const int L1GtPsbWord::NumberBData;



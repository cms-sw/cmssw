/**
 * \class L1GtfeWord
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
#include "DataFormats/L1GlobalTrigger/interface/L1GtfeWord.h"

// system include files

// user include files
#include "FWCore/Utilities/interface/EDMException.h"


// constructors

// empty constructor, all members set to zero;
L1GtfeWord::L1GtfeWord() {

    m_boardId = 0;
    m_recordLength = 0;  
    m_bxNr = 0; 
    m_setupVersion = 0; 
    m_activeBoards = 0;
    m_totalTriggerNr = 0;
    
    for (int iB = 0; iB < NumberGpsTimes; ++iB) {
        m_gpsTime[iB] = 0;            
    }

}

// constructor from unpacked values;
L1GtfeWord::L1GtfeWord(
    uint16_t boardIdValue,
    uint16_t recordLengthValue,  
    uint16_t bxNrValue, 
    uint32_t setupVersionValue, 
    uint16_t activeBoardsValue,
    uint32_t totalTriggerNrValue,
    uint16_t gpsTimeValue[NumberGpsTimes] ) 
{

    m_boardId = boardIdValue;
    m_recordLength = recordLengthValue;  
    m_bxNr = bxNrValue; 
    m_setupVersion = setupVersionValue; 
    m_activeBoards = activeBoardsValue;
    m_totalTriggerNr = totalTriggerNrValue;
    
    for (int iB = 0; iB < NumberGpsTimes; ++iB) {
        m_gpsTime[iB] = gpsTimeValue[iB];                          
    }

}

// destructor
L1GtfeWord::~L1GtfeWord() {
}

// equal operator
bool L1GtfeWord::operator==(const L1GtfeWord& result) const {

    if(m_boardId != result.m_boardId) return false;
    if(m_recordLength != result.m_recordLength) return false;  
    if(m_bxNr != result.m_bxNr) return false; 
    if(m_setupVersion != result.m_setupVersion) return false; 
    if(m_activeBoards != result.m_activeBoards) return false;
    if(m_totalTriggerNr != result.m_totalTriggerNr) return false;
    
    for (int iB = 0; iB < NumberGpsTimes; ++iB) {
        if(m_gpsTime[iB] != result.m_gpsTime[iB]) return false;                          
    }


    // all members identical
    return true;
    
}

// unequal operator
bool L1GtfeWord::operator!=(const L1GtfeWord& result) const{
    
    return !( result == *this);
    
}

// methods

// get/set GPS time
const uint16_t L1GtfeWord::gpsTime(int iB) const {

    if (iB < 0 || iB > NumberGpsTimes) {
        throw cms::Exception("GpsTimeIndexError")
            << "\nError: index for GPS time array out of range. Allowed range: [0, "
            << NumberGpsTimes << ") " << std::endl;
        
    } else {
         return m_gpsTime[iB];         
    }     
     
}
void L1GtfeWord::setGpsTime(uint16_t gpsTimeVal, int iB) { 

    if (iB < 0 || iB > NumberGpsTimes) {
        throw cms::Exception("GpsTimeIndexError")
            << "\nError: index for GPS time array out of range. Allowed range: [0, "
            << NumberGpsTimes << ") " << std::endl;
        
    } else {
        m_gpsTime[iB] = gpsTimeVal; 
    }     
    
}

// static class members

const int L1GtfeWord::NumberGpsTimes;


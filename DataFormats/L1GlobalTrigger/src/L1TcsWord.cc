/**
 * \class L1TcsWord
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
#include "DataFormats/L1GlobalTrigger/interface/L1TcsWord.h"

// system include files

// user include files
//   base class

// forward declarations

// constructors

// empty constructor, all members set to zero;
L1TcsWord::L1TcsWord() {
    
    m_daqNr = 0;
    m_triggerType = 0;
    m_status = 0;
    m_bxNr = 0;
    m_partTrigNr = 0;
    m_partRunNr = 0;
    m_eventNr = 0; 
    m_assignedPartitions = 0; 
    m_orbitNr = 0;
    
}

// constructor from unpacked values;
L1TcsWord::L1TcsWord(
    uint16_t daqNrValue,
    uint16_t triggerTypeValue,
    uint16_t statusValue,
    uint16_t bxNrValue,
    uint32_t partTrigNrValue,
    uint32_t partRunNrValue,
    uint32_t eventNrValue, 
    uint32_t assignedPartitionsValue, 
    uint32_t orbitNrValue ) 
{
    
    m_daqNr = daqNrValue;
    m_triggerType = triggerTypeValue;
    m_status = statusValue;
    m_bxNr = bxNrValue;
    m_partTrigNr = partTrigNrValue;
    m_partRunNr = partRunNrValue;
    m_eventNr = eventNrValue; 
    m_assignedPartitions = assignedPartitionsValue; 
    m_orbitNr = orbitNrValue;
    
}

// destructor
L1TcsWord::~L1TcsWord() {
}

// equal operator
bool L1TcsWord::operator==(const L1TcsWord& result) const {

    if (m_daqNr != result.m_daqNr) return false;
    if (m_triggerType != result.m_triggerType) return false;
    if (m_status != result.m_status) return false;
    if (m_bxNr != result.m_bxNr) return false;
    if (m_partTrigNr != result.m_partTrigNr) return false;
    if (m_partRunNr != result.m_partRunNr) return false;
    if (m_eventNr != result.m_eventNr) return false; 
    if (m_assignedPartitions != result.m_assignedPartitions) return false; 
    if (m_orbitNr != result.m_orbitNr) return false;

    // all members identical
    return true;
    
}

// unequal operator
bool L1TcsWord::operator!=(const L1TcsWord& result) const{
    
    return !( result == *this);
    
}


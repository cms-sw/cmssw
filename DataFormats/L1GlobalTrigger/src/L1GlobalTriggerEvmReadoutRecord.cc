/**
 * \class L1GlobalTriggerEvmReadoutRecord
 * 
 * 
 * 
 * Description: see header file 
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete   - HEPHY Vienna - CMSSW version 
 * 
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"

// system include files
#include <iostream>
#include <iomanip>
#include <bitset>
#include <boost/cstdint.hpp>


// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GtfeWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1TcsWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtFdlWord.h"

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// constructors
L1GlobalTriggerEvmReadoutRecord::L1GlobalTriggerEvmReadoutRecord() {

    m_gtfeWord = L1GtfeWord();
    m_tcsWord  = L1TcsWord();
    
    // reserve just one L1GtFdlWord
    m_gtFdlWord.reserve(1);
    m_gtFdlWord.assign(1, L1GtFdlWord());
         
}

L1GlobalTriggerEvmReadoutRecord::L1GlobalTriggerEvmReadoutRecord(int NumberBxInEvent) {

    m_gtfeWord = L1GtfeWord();
    m_tcsWord  = L1TcsWord();
    
    m_gtFdlWord.reserve(NumberBxInEvent);
    m_gtFdlWord.assign(NumberBxInEvent, L1GtFdlWord());

    for (int iBx = 0; iBx < NumberBxInEvent; ++iBx) {  // TODO review here after hw discussion
        m_gtFdlWord[iBx].setBxInEvent(iBx);
    }        
    
}

// copy constructor
L1GlobalTriggerEvmReadoutRecord::L1GlobalTriggerEvmReadoutRecord(
    const L1GlobalTriggerEvmReadoutRecord& result) {

    m_gtfeWord = result.m_gtfeWord;
    m_tcsWord = result.m_tcsWord;
    m_gtFdlWord = result.m_gtFdlWord;
        
 }

// destructor
L1GlobalTriggerEvmReadoutRecord::~L1GlobalTriggerEvmReadoutRecord() {
    
    
}
      
// assignment operator
L1GlobalTriggerEvmReadoutRecord& L1GlobalTriggerEvmReadoutRecord::operator=(
    const L1GlobalTriggerEvmReadoutRecord& result) {

    if ( this != &result ) {

        m_gtfeWord = result.m_gtfeWord;
        m_tcsWord  = result.m_tcsWord;
        m_gtFdlWord  = result.m_gtFdlWord;
        
    }
    
    return *this;
    
 }
  
// equal operator
bool L1GlobalTriggerEvmReadoutRecord::operator==(
    const L1GlobalTriggerEvmReadoutRecord& result) const {

    if (m_gtfeWord != result.m_gtfeWord) return false;
    if (m_tcsWord  != result.m_tcsWord)  return false;
    
    if (m_gtFdlWord  != result.m_gtFdlWord)  return false;

    // all members identical
    return true;
    
}

// unequal operator
bool L1GlobalTriggerEvmReadoutRecord::operator!=(
    const L1GlobalTriggerEvmReadoutRecord& result) const{
    
    return !( result == *this);
    
}
// methods


// get Global Trigger decision and the decision word
//    general bxInEvent 
const bool L1GlobalTriggerEvmReadoutRecord::decision(unsigned int bxInEventValue) const {

    for (std::vector<L1GtFdlWord>::const_iterator itBx = m_gtFdlWord.begin(); 
        itBx != m_gtFdlWord.end(); ++itBx) {
        
        if ( (*itBx).bxInEvent() == bxInEventValue ) {
            return (*itBx).finalOR();
        }               
    }
    
    // if bunch cross not found, throw exception (action: SkipEvent) 
    
    throw cms::Exception("NotFound")
        << "\nError: requested GtFdlWord for bx = " << bxInEventValue << " does not exist.\n"
        << "Can not return global decision for this bx!\n"
        << std::endl;

    return false;          
}

const L1GlobalTriggerEvmReadoutRecord::DecisionWord 
    L1GlobalTriggerEvmReadoutRecord::decisionWord(unsigned int bxInEventValue) const {

    for (std::vector<L1GtFdlWord>::const_iterator itBx = m_gtFdlWord.begin(); 
        itBx != m_gtFdlWord.end(); ++itBx) {
        
        if ( (*itBx).bxInEvent() == bxInEventValue ) {
            return (*itBx).gtDecisionWord();
        }               
    }
    
    // if bunch cross not found, throw exception (action: SkipEvent)
    
    throw cms::Exception("NotFound")
        << "\nError: requested GtFdlWord for bx = " << bxInEventValue << " does not exist.\n"
        << "Can not return decision word for this bx!\n"
        << std::endl;

    DecisionWord dW; // empty; it does not arrive here
    return dW;          
}

//    bxInEvent = 0  
const bool L1GlobalTriggerEvmReadoutRecord::decision() const {
    
    unsigned int bxInEventL1Accept = 0;
    return decision(bxInEventL1Accept);
}

const L1GlobalTriggerEvmReadoutRecord::DecisionWord 
    L1GlobalTriggerEvmReadoutRecord::decisionWord() const { 
    
    unsigned int bxInEventL1Accept = 0;
    return decisionWord(bxInEventL1Accept);
}
  

// set global decision and the decision word
//    general 
void L1GlobalTriggerEvmReadoutRecord::setDecision(bool t, unsigned int bxInEventValue) { 

    for (std::vector<L1GtFdlWord>::iterator itBx = m_gtFdlWord.begin(); 
        itBx != m_gtFdlWord.end(); ++itBx) {
        
        if ( (*itBx).bxInEvent() == bxInEventValue ) {
            (*itBx).setFinalOR(static_cast<uint16_t> (t)); // TODO FIXME when manipulating partitions
        }               
    }
    
    // if bunch cross not found, throw exception (action: SkipEvent)
    
    throw cms::Exception("NotFound")
        << "\nError: requested GtFdlWord for bx = " << bxInEventValue << " does not exist.\n"
        << "Can not set global decision for this bx!\n"
        << std::endl;

}

void L1GlobalTriggerEvmReadoutRecord::setDecisionWord(
    const L1GlobalTriggerEvmReadoutRecord::DecisionWord& decisionWordValue, 
    unsigned int bxInEventValue) {

    for (std::vector<L1GtFdlWord>::iterator itBx = m_gtFdlWord.begin(); 
        itBx != m_gtFdlWord.end(); ++itBx) {
        
        if ( (*itBx).bxInEvent() == bxInEventValue ) {
            (*itBx).setGtDecisionWord (decisionWordValue);
        }               
    }
    
    // if bunch cross not found, throw exception (action: SkipEvent)
    
    throw cms::Exception("NotFound")
        << "\nError: requested GtFdlWord for bx = " << bxInEventValue << " does not exist.\n"
        << "Can not set decision word for this bx!\n"
        << std::endl;

}

//    bxInEvent = 0  
void L1GlobalTriggerEvmReadoutRecord::setDecision(bool t) { 

    unsigned int bxInEventL1Accept = 0;
    setDecision(t, bxInEventL1Accept);
}

void L1GlobalTriggerEvmReadoutRecord::setDecisionWord(
    const L1GlobalTriggerEvmReadoutRecord::DecisionWord& decisionWordValue) {

    unsigned int bxInEventL1Accept = 0;
    setDecisionWord(decisionWordValue, bxInEventL1Accept);

}

// print global decision and algorithm decision word
void L1GlobalTriggerEvmReadoutRecord::printGtDecision(
    std::ostream& myCout, unsigned int bxInEventValue
    ) const {
        
    for (std::vector<L1GtFdlWord>::const_iterator itBx = m_gtFdlWord.begin(); 
        itBx != m_gtFdlWord.end(); ++itBx) {
        
        if ( (*itBx).bxInEvent() == bxInEventValue ) {

            myCout << "\nL1 Global Trigger Record: " << std::endl;
        
            myCout << "\t Bunch cross " << bxInEventValue 
                << std::endl
                << "\t Global Decision = " << std::setw(5) << (*itBx).globalDecision()
                 << std::endl 
                << "\t Decision word (bitset style) = "; 
            
            (*itBx).printGtDecisionWord(myCout);
        }               
    }      

    myCout << std::endl;
    
}

void L1GlobalTriggerEvmReadoutRecord::printGtDecision(std::ostream& myCout) const {
    
    unsigned int bxInEventL1Accept = 0;
    printGtDecision(myCout, bxInEventL1Accept);
        
}

// print technical trigger word (reverse order for vector<bool>)
void L1GlobalTriggerEvmReadoutRecord::printTechnicalTrigger(
    std::ostream& myCout, unsigned int bxInEventValue
    ) const {
        
    for (std::vector<L1GtFdlWord>::const_iterator itBx = m_gtFdlWord.begin(); 
        itBx != m_gtFdlWord.end(); ++itBx) {
        
        if ( (*itBx).bxInEvent() == bxInEventValue ) {

            myCout << "\nL1 Global Trigger Record: " << std::endl;
        
            myCout << "\t Bunch cross " << bxInEventValue 
                << std::endl
                << "\t Technical Trigger word (bitset style) = "; 
            
            (*itBx).printGtTechnicalTriggerWord(myCout);
        }               
    }      

    myCout << std::endl;
    
}

void L1GlobalTriggerEvmReadoutRecord::printTechnicalTrigger(std::ostream& myCout) const {
    
    unsigned int bxInEventL1Accept = 0;
    printTechnicalTrigger(myCout, bxInEventL1Accept);
        
}

// get/set hardware-related words

// get / set GTFE word (record) in the GT readout record
const L1GtfeWord L1GlobalTriggerEvmReadoutRecord::gtfeWord() const {

    return m_gtfeWord;
    
}

void L1GlobalTriggerEvmReadoutRecord::setGtfeWord(const L1GtfeWord& gtfeWordValue) {

    m_gtfeWord = gtfeWordValue;

}

// get / set TCS word (record) in the GT readout record
const L1TcsWord L1GlobalTriggerEvmReadoutRecord::tcsWord() const {

    return m_tcsWord;
    
}

void L1GlobalTriggerEvmReadoutRecord::setTcsWord(const L1TcsWord& tcsWordValue) {

    m_tcsWord = tcsWordValue;

}

// get / set FDL word (record) in the GT readout record
const L1GtFdlWord L1GlobalTriggerEvmReadoutRecord::gtFdlWord(unsigned int bxInEventValue) const {

    for (std::vector<L1GtFdlWord>::const_iterator itBx = m_gtFdlWord.begin(); 
        itBx != m_gtFdlWord.end(); ++itBx) {
        
        if ( (*itBx).bxInEvent() == bxInEventValue ) {
            return (*itBx);
        }               
    }      
    
    // if bunch cross not found, throw exception (action: SkipEvent) 
    
    throw cms::Exception("NotFound")
        << "\nError: requested L1GtFdlWord for bx = " << bxInEventValue << " does not exist.\n"
        << std::endl;

    // return empty record - actually does not arrive here 
    return L1GtFdlWord();          
    
}

const L1GtFdlWord L1GlobalTriggerEvmReadoutRecord::gtFdlWord() const {

    unsigned int bxInEventL1Accept = 0;
    return gtFdlWord(bxInEventL1Accept);
}

void L1GlobalTriggerEvmReadoutRecord::setGtFdlWord(
    const L1GtFdlWord& gtFdlWordValue, unsigned int bxInEventValue) {

    // if a L1GtFdlWord exists for bxInEventValue, replace it
    for (std::vector<L1GtFdlWord>::iterator itBx = m_gtFdlWord.begin(); 
        itBx != m_gtFdlWord.end(); ++itBx) {
        
        if ( (*itBx).bxInEvent() == bxInEventValue ) {
            *itBx = gtFdlWordValue;
            LogDebug("L1GlobalTriggerEvmReadoutRecord") 
                << "Replacing L1GtFdlWord for bunch bx = " << bxInEventValue << "\n" 
                << std::endl;
            return;
        }               
    }
    
    // if bunch cross not found, throw exception (action: SkipEvent) 
    // all L1GtFdlWord are created in the record constructor for allowed bunch crosses
    
    throw cms::Exception("NotFound")
        << "\nError: Cannot set L1GtFdlWord for bx = " << bxInEventValue 
        << std::endl;

}

void L1GlobalTriggerEvmReadoutRecord::setGtFdlWord(const L1GtFdlWord& gtFdlWordValue) {

    unsigned int bxInEventL1Accept = 0;
    setGtFdlWord(gtFdlWordValue, bxInEventL1Accept);
    
}


// other methods

// clear the record
void L1GlobalTriggerEvmReadoutRecord::reset() {

    // TODO FIXME clear GTFE, TCS, FDL?
        
} 
  
// output stream operator
std::ostream& operator<<(std::ostream& s, const L1GlobalTriggerEvmReadoutRecord& result) {
    // TODO FIXME put together all prints
    s << "Not available yet - sorry";
        
  return s;
    
}

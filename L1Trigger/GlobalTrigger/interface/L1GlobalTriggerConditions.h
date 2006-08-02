#ifndef GlobalTrigger_L1GlobalTriggerConditions_h
#define GlobalTrigger_L1GlobalTriggerConditions_h

/**
 * \class L1GlobalTriggerConditions
 * 
 * 
 * 
 * Description: Base class for XML particle templates
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: M.Eder, H. Rohringer - HEPHY Vienna - ORCA version 
 * \author: Vasile Mihai Ghete   - HEPHY Vienna - CMSSW version 
 * 
 * $Date:$
 * $Revision:$
 *
 */

// system include files
#include <iostream>
#include <iomanip>
#include <string>

// user include files 
#include "DataFormats/L1GlobalTrigger/interface/L1TriggerObject.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// forward declarations
//class L1TriggerObject;

// class interface
class L1GlobalTriggerConditions 
{
    
public:

    /// constructor
    L1GlobalTriggerConditions(const std::string& );

    /// copy constructor
    L1GlobalTriggerConditions(L1GlobalTriggerConditions& );
    
    /// destructor
    virtual ~L1GlobalTriggerConditions();
    
public:
    
    inline bool getGeEq() const { return p_ge_eq; }
    inline void setGeEq(bool ge_eq) { p_ge_eq = ge_eq; }

    inline const std::string& getName() const { return p_name; }

    inline bool getLastResult() const { return p_lastresult; }

    inline int getOutputPin() const { return p_outputpin; }
    inline void setOutputPin(int pin) { p_outputpin = pin; }

//    /// get candidates
//    virtual L1TriggerObject* getCandidate( int indexCand ) const = 0;

    /// call blockCondition and save last result
    inline const bool blockCondition_sr() { return p_lastresult = blockCondition(); }
  
    /// the core function to check if the condition matches
    virtual const bool blockCondition() const = 0;
  
    /// print thresholds
    virtual void printThresholds() const = 0;

    /// get numeric expression
    virtual std::string getNumericExpression() { if (p_lastresult) { return "1"; } else { return "0";} }

protected:

    /// the name of the condition
    std::string p_name;

    /// the last result of blockCondition()
    bool p_lastresult;
    
    /// the value of the ge_eq flag
    bool p_ge_eq;

    ///
    int p_outputpin;    

protected:

    /// check if a value is greater than a threshold or 
    /// greater equal depending on the value of the geeq flag  
    template<class Type1, class Type2>
        const bool checkThreshold(Type1 const &threshold, Type2 const &value) const;
     
    ///check if a bit with a given number is set in a mask
    template<class Type1>
        const bool checkBit(Type1 const &mask, unsigned int bitNumber) const;
        
   

};
 
// define templated methods here, otherwise one can not link 

// check if a value is greater than a threshold or 
// greater equal depending on the value of the geeq flag  
template<class Type1, class Type2>
    const bool L1GlobalTriggerConditions::checkThreshold(
    Type1 const &threshold, Type2 const &value) const {

    if (p_ge_eq) {
        if ( value > 0 ) {
            edm::LogVerbatim("L1GlobalTriggerConditions") 
                << " p_ge_eq threshold CHECK " 
                << " hex " << std::hex << threshold << " " << value << " DEC " << std::dec 
                << threshold << " "  << value << std::endl;
        }
        
        if ( value >= threshold ) {
            edm::LogVerbatim("L1GlobalTriggerConditions") << " CHECK  triggered p_ge_eq " 
                << std::dec << std::endl;
        }
        
        return (value >= threshold);
        
    } else {
        edm::LogVerbatim("L1GlobalTriggerConditions")  
            << " p       threshold CHECK" 
            << " hex " << threshold << " " << value << " dec " << std::dec 
            << threshold << " " << value << std::endl;
        if ( value > threshold ) {
            edm::LogVerbatim("L1GlobalTriggerConditions") << " CHECK triggered " << std::endl;
        }
        
        return (value > threshold);
    }
}


// check if a bit with a given number is set in a mask
template<class Type1>
    const bool L1GlobalTriggerConditions::checkBit(
    Type1 const &mask, unsigned int bitNumber) const {

    if (bitNumber >= 64) return false; // TODO 64 as static or parameter

    u_int64_t oneBit = 1;
    oneBit <<= bitNumber;  
        
    bool mask1one = mask & oneBit;
    
    edm::LogVerbatim("L1GlobalTriggerConditions") 
        << " checkBit " 
        << &mask << " " << mask << " " << oneBit << " " << bitNumber 
        << " result " << mask1one 
        << std::endl;
    edm::LogVerbatim("L1GlobalTriggerConditions") 
        << " checkBit hex " 
        << std::hex << mask <<" " << std::dec 
        << std::endl;
    
    if (mask1one == 0 ) { 
        edm::LogVerbatim("L1GlobalTriggerConditions") << " zero result " << std::endl;
        edm::LogVerbatim("L1GlobalTriggerConditions") << std::hex << mask << std::endl;
        edm::LogVerbatim("L1GlobalTriggerConditions") << std::hex << oneBit  << std::endl;
        edm::LogVerbatim("L1GlobalTriggerConditions") << std::dec << std::endl;
    }
    
    return (mask & oneBit);
}

#endif

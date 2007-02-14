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
 * $Date$
 * $Revision$
 *
 */

// system include files
#include <iostream>
#include <iomanip>

#include <string>
#include <vector>

// user include files 
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// forward declarations
class L1GlobalTrigger;

// class interface
class L1GlobalTriggerConditions 
{
    
public:

    /// constructor
    L1GlobalTriggerConditions(const L1GlobalTrigger&, const std::string& );

    /// copy constructor
    L1GlobalTriggerConditions(L1GlobalTriggerConditions& );
    
    /// destructor
    virtual ~L1GlobalTriggerConditions();

public:
    
    inline bool getGeEq() const { return p_ge_eq; }
    inline void setGeEq(bool ge_eq) { p_ge_eq = ge_eq; }

    inline const std::string& getName() const { return p_name; }

    inline bool getLastResult() const { return p_lastresult; }

    /// get / set output pins
    inline int getOutputPin() const { return p_outputpin; }
    inline void setOutputPin(int pin) { p_outputpin = pin; }

    /// get / set algorithm number 
    // depends on the order (connection) of the condition chips in the hardware
    inline int getAlgoNumber() const { return p_algoNumber; }
    inline void setAlgoNumber(int algoNumber) { p_algoNumber = algoNumber; }

    /// call blockCondition and save last result
    inline const bool blockCondition_sr() { return p_lastresult = blockCondition(); }
  
    /// the core function to check if the condition matches
    virtual const bool blockCondition() const = 0;
  
    /// print thresholds
    virtual void printThresholds(std::ostream& myCout) const = 0;

    /// get logical expression
    virtual std::string getLogicalExpression() { return p_name; }

    /// get numeric expression
    virtual std::string getNumericExpression() { if (p_lastresult) { return "1"; } else { return "0";} }

    /// get the vector of combinations for the algorithm
    virtual std::vector<CombinationsInCond> getCombinationVector();

    /// get all the object combinations evaluated to true in the condition
    inline CombinationsInCond* getCombinationsInCond() const { return p_combinationsInCond; }

protected:

    /// the name of the condition
    std::string p_name;

    /// the last result of blockCondition()
    bool p_lastresult;
    
    /// the value of the ge_eq flag
    bool p_ge_eq;

    /// output pin on condition chip
    int p_outputpin;    

    /// algorithm number (bit number in decision word)
    int p_algoNumber;    
    
    /// store all the object combinations evaluated to true in the condition     
    CombinationsInCond* p_combinationsInCond;


protected:

    /// check if a value is greater than a threshold or 
    /// greater equal depending on the value of the geeq flag  
    template<class Type1, class Type2>
        const bool checkThreshold(Type1 const &threshold, Type2 const &value) const;
     
    ///check if a bit with a given number is set in a mask
    template<class Type1>
        const bool checkBit(Type1 const &mask, unsigned int bitNumber) const;
        
   
    const L1GlobalTrigger& m_GT;

};
 
// define templated methods here, otherwise one can not link 

// check if a value is greater than a threshold or 
// greater equal depending on the value of the geeq flag  
template<class Type1, class Type2>
    const bool L1GlobalTriggerConditions::checkThreshold(
    Type1 const &threshold, Type2 const &value) const {

    if (p_ge_eq) {
        if ( value > 0 ) {
            LogTrace("L1GlobalTriggerConditions") 
                << "  p_ge_eq threshold check:" 
                << "\n    hex: " << std::hex << "threshold = " << threshold << " value = " << value 
                << "\n    dec: " << std::dec << "threshold = " << threshold << " value = " << value
                << std::endl;
        }
        
        if ( value >= threshold ) {
            LogTrace("L1GlobalTriggerConditions") << "    p_ge_eq: value >= threshold" 
                << std::endl;
        }
        
        return (value >= threshold);
        
    } else {
        LogTrace("L1GlobalTriggerConditions") 
            << "  p threshold check:" 
            << "\n    hex: " << std::hex << "threshold = " << threshold << " value = " << value 
            << "\n    dec: " << std::dec << "threshold = " << threshold << " value = " << value
            << std::endl;
        if ( value > threshold ) {
            LogTrace("L1GlobalTriggerConditions") << "    p: value > threshold" 
                << std::endl;
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
        
    LogTrace("L1GlobalTriggerConditions") 
        << "    checkBit " 
        << "\n     mask address = " << &mask
        << std::dec  
        << "\n     dec: mask = " << mask << " oneBit = " << oneBit << " bitNumber = " << bitNumber 
        << std::hex 
        << "\n     hex: mask = " << mask << " oneBit = " << oneBit << " bitNumber = " << bitNumber
        << std::dec 
        << "\n     mask & oneBit result = " << bool ( mask & oneBit ) 
        << std::endl;
    
    return (mask & oneBit);
}

#endif

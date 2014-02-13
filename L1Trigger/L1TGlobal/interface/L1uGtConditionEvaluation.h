#ifndef GlobalTrigger_L1uGtConditionEvaluation_h
#define GlobalTrigger_L1uGtConditionEvaluation_h

/**
 * \class L1uGtConditionEvaluation
 *
 *
 * Description: Base class for evaluation of the L1 Global Trigger object templates.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete   - HEPHY Vienna
 *
 *
 */

// system include files
#include <iostream>

#include <string>
#include <vector>

#include <boost/cstdint.hpp>

// user include files

//   base class

//
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// forward declarations

namespace l1t {

// class interface
class L1uGtConditionEvaluation
{

public:

    /// constructor
    L1uGtConditionEvaluation()  :
    m_condMaxNumberObjects(0),
    m_condLastResult(false),
    m_verbosity(0) {}


    /// destructor
  virtual ~L1uGtConditionEvaluation(){}

public:

    /// get / set the maximum number of objects received for the
    /// evaluation of the condition
    inline int condMaxNumberObjects() const {
        return m_condMaxNumberObjects;
    }

    inline void setCondMaxNumberObjects(int condMaxNumberObjectsValue) {
        m_condMaxNumberObjects = condMaxNumberObjectsValue;
    }

    /// get the latest result for the condition
    inline bool condLastResult() const {
        return m_condLastResult;
    }

    /// call evaluateCondition and save last result
    inline void evaluateConditionStoreResult(const int bxEval) {
        m_condLastResult = evaluateCondition(bxEval);
    }

    /// the core function to check if the condition matches
    virtual const bool evaluateCondition(const int bxEval) const = 0;

    /// get numeric expression
    virtual std::string getNumericExpression() const {
        if (m_condLastResult) {
            return "1";
        }
        else {
            return "0";
        }
    }

    /// get all the object combinations evaluated to true in the condition
    inline CombinationsInCond const & getCombinationsInCond() const {
        return m_combinationsInCond;
    }


    /// print condition
    virtual void print(std::ostream& myCout) const;

    inline void setVerbosity(const int verbosity) {
        m_verbosity = verbosity;
    }

protected:

    /// get all the object combinations (to fill it...)
    inline CombinationsInCond& combinationsInCond() const {
        return m_combinationsInCond;
    }

    /// check if a value is greater than a threshold or
    /// greater-or-equal depending on the value of the condGEqValue flag
    template<class Type1, class Type2> const bool checkThreshold(const Type1& threshold,
        const Type2& value, const bool condGEqValue) const;

    /// check if a bit with a given number is set in a mask
    template<class Type1> const bool checkBit(const Type1& mask, const unsigned int bitNumber) const;

    /// check if a value is in a given range and outside of a veto range
    template<class Type1> const bool checkRange(const unsigned int bitNumber, 
						const Type1& beginR, const Type1& endR, 
						const Type1& beginVetoR, const Type1& endVetoR ) const;


protected:

    /// maximum number of objects received for the evaluation of the condition
    /// usually retrieved from event setup
    int m_condMaxNumberObjects;

    /// the last result of evaluateCondition()
    bool m_condLastResult;

    /// store all the object combinations evaluated to true in the condition
    mutable CombinationsInCond m_combinationsInCond;

    /// verbosity level
    int m_verbosity;

};

// define templated methods

// check if a value is greater than a threshold or
// greater-or-equal depending on the value of the condGEqValue flag
template<class Type1, class Type2> const bool L1uGtConditionEvaluation::checkThreshold(
    const Type1& threshold, const Type2& value, const bool condGEqValue) const {

    //if (value > 0) {
    //    LogTrace("L1GlobalTrigger") << "  threshold check for condGEqValue = "
    //        << condGEqValue << "\n    hex: " << std::hex << "threshold = " << threshold
    //        << " value = " << value << "\n    dec: " << std::dec << "threshold = " << threshold
    //        << " value = " << value << std::endl;
    //}

    if (condGEqValue) {
        if (value >= threshold) {

            //LogTrace("L1GlobalTrigger") << "    condGEqValue: value >= threshold"
            //    << std::endl;

            return true;
        }

        return false;

    }
    else {

        if (value == threshold) {

            //LogTrace("L1GlobalTrigger") << "    condGEqValue: value = threshold"
            //    << std::endl;

            return true;
        }

        return false;
    }
}

// check if a bit with a given number is set in a mask
template<class Type1> const bool L1uGtConditionEvaluation::checkBit(const Type1& mask,
    const unsigned int bitNumber) const {

    boost::uint64_t oneBit = 1ULL;

    if (bitNumber >= (sizeof(oneBit)*8)) {

        if (m_verbosity) {

            LogTrace("L1GlobalTrigger")
                << "    checkBit " << "\n     Bit number = "
                << bitNumber << " larger than maximum allowed " << sizeof ( oneBit ) * 8
                << std::endl;
        }

        return false;
    }

    oneBit <<= bitNumber;

    //LogTrace("L1GlobalTrigger") << "    checkBit " << "\n     mask address = " << &mask
    //    << std::dec << "\n     dec: " << "mask = " << mask << " oneBit = " << oneBit
    //    << " bitNumber = " << bitNumber << std::hex << "\n     hex: " << "mask = " << mask
    //    << " oneBit = " << oneBit << " bitNumber = " << bitNumber << std::dec
    //    << "\n     mask & oneBit result = " << bool ( mask & oneBit ) << std::endl;

    return (mask & oneBit);
}


/// check if a value is in a given range and outside of a veto range
template<class Type1> const bool L1uGtConditionEvaluation::checkRange(const unsigned int bitNumber, 
								      const Type1& beginR, const Type1& endR, 
								      const Type1& beginVetoR, const Type1& endVetoR ) const {

  // set condtion to true if beginR==endR = default -1
  if( beginR==endR && beginR==-1 ){
    return true;
  }

/*   LogDebug("l1t|Global")  */
/*     << "\n l1t::L1uGtConditionEvaluation"  */
/*     << "\n\t bitNumber = " << bitNumber */
/*     << "\n\t beginR = " << beginR */
/*     << "\n\t endR   = " << endR */
/*     << "\n\t beginVetoR = " << beginVetoR */
/*     << "\n\t endVetoR   = " << endVetoR */
/*     << std::endl; */

  // check if value is in range
  // for begin <= end takes [begin, end]
  // for begin >= end takes [begin, end] over zero angle!
  if( endR >= beginR ){
    if( !( bitNumber>=beginR && bitNumber<=endR ) ){
      return false;
    }
    else if( beginVetoR==endVetoR ){
      return true;
    }
    else {
      if( endVetoR >= beginVetoR ){
	if( !( bitNumber<=beginVetoR && bitNumber>=endVetoR ) ){
	  return false;
	}
	else{
	  return true;
	}
      }
      else{ // go over zero angle!!
      	if( !( bitNumber<=beginVetoR || bitNumber>=endVetoR ) ){
	  return false;
	}
	else{
	  return true;
	}
      }
    }
  }
  else { // go over zero angle!!
    if( !( bitNumber>=beginR || bitNumber<=endR ) ){
      return false;
    }
    else if( beginVetoR==endVetoR ){
      return true;
    }
    else {
      if( endVetoR >= beginVetoR ){
	if( !( bitNumber<=beginVetoR && bitNumber>=endVetoR ) ){
	  return false;
	}
	else{
	  return true;
	}
      }
      else{ // go over zero angle!!
      	if( !( bitNumber<=beginVetoR || bitNumber>=endVetoR ) ){
	  return false;
	}
	else{
	  return true;
	}
      }
    }
  }


  LogDebug("l1t|Global") << "=====> HELP!! I'm trapped and I cannot escape! AHHHHHH" << std::endl;

/*   // DMP Not sure about this, will check with hardware */
/*   // check to make sure beginRange comes before endRange */
/*   if( beginR>endR ){ */
/*     LogTrace("l1t|Global") << " ====> WARNING: range begin = " << beginR << " < end = " << endR << std::endl; */
/*     return false; */
/*   } */

/*   // DMP Not sure about this, will check with hardware */
/*   // check to make sure beginVetoRange comes before endVetoRange */
/*   if( beginVetoR>endVetoR ){ */
/*     LogTrace("l1t|Global") << " ====> WARNING: range veto begin = " << beginR << " < end = " << endR << std::endl; */
/*     return false; */
/*   } */
   
}



}
#endif

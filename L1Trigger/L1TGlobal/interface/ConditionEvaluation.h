#ifndef GlobalTrigger_ConditionEvaluation_h
#define GlobalTrigger_ConditionEvaluation_h

/**
 * \class ConditionEvaluation
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
class ConditionEvaluation
{

public:

    /// constructor
    ConditionEvaluation()  :
    m_condMaxNumberObjects(0),
    m_condLastResult(false),
    m_verbosity(0) {}


    /// destructor
    virtual ~ConditionEvaluation(){}

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
        const Type2& value, bool condGEqValue) const;

    /// check if a bit with a given number is set in a mask
    template<class Type1> const bool checkBit(const Type1& mask, const unsigned int bitNumber) const;

    /// check if a value is in a given range and outside of a veto range
    template<class Type1> const bool checkRangeEta(const unsigned int bitNumber, 
						   const Type1& beginR, const Type1& endR, 
						   const Type1& beginVetoR, const Type1& endVetoR,
						   const unsigned int nEtaBits ) const;

    /// check if a value is in a given range and outside of a veto range
    template<class Type1> const bool checkRangePhi(const unsigned int bitNumber, 
						   const Type1& beginR, const Type1& endR, 
						   const Type1& beginVetoR, const Type1& endVetoR ) const;


    /// check if a value is in a given range 
    template<class Type1> const bool checkRangeDeltaEta(const unsigned int obj1Eta, const unsigned int obj2Eta, 
							const Type1& lowerR, const Type1& upperR,
							const unsigned int nEtaBits ) const;

    /// check if a value is in a given range 
    template<class Type1> const bool checkRangeDeltaPhi(const unsigned int obj1Phi, const unsigned int obj2Phi,
							const Type1& lowerR, const Type1& upperR ) const;


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
template<class Type1, class Type2> const bool ConditionEvaluation::checkThreshold(
    const Type1& threshold, const Type2& value, const bool condGEqValue) const {

    //if (value > 0) {
    //    LogTrace("L1GlobalTrigger") << "  threshold check for condGEqValue = "
    //        << condGEqValue << "\n    hex: " << std::hex << "threshold = " << threshold
    //        << " value = " << value << "\n    dec: " << std::dec << "threshold = " << threshold
    //        << " value = " << value << std::endl;
    //}

    if (condGEqValue) {
        if (value >= (Type2) threshold) {

            //LogTrace("L1GlobalTrigger") << "    condGEqValue: value >= threshold"
            //    << std::endl;

            return true;
        }

        return false;

    }
    else {

        if (value == (Type2) threshold) {

            //LogTrace("L1GlobalTrigger") << "    condGEqValue: value = threshold"
            //    << std::endl;

            return true;
        }

        return false;
    }
}

// check if a bit with a given number is set in a mask
template<class Type1> const bool ConditionEvaluation::checkBit(const Type1& mask,
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
 template<class Type1> const bool ConditionEvaluation::checkRangeEta(const unsigned int bitNumber, 
								     const Type1& beginR, const Type1& endR, 
								     const Type1& beginVetoR, const Type1& endVetoR,
								     const unsigned int nEtaBits) const {

  // set condtion to true if beginR==endR = default -1
  if( beginR==endR && beginR==(Type1)-1 ){
    return true;
  }

  unsigned int diff1 = endR - beginR;
  unsigned int diff2 = bitNumber - beginR;
  unsigned int diff3 = endR - bitNumber;

  bool cond1 = ( (diff1>>nEtaBits) & 1 ) ? false : true;
  bool cond2 = ( (diff2>>nEtaBits) & 1 ) ? false : true;
  bool cond3 = ( (diff3>>nEtaBits) & 1 ) ? false : true;

  LogDebug("l1t|Global")
    << "\n l1t::ConditionEvaluation"
    << "\n\t bitNumber = " << bitNumber
    << "\n\t beginR = " << beginR
    << "\n\t endR   = " << endR
    << "\n\t beginVetoR = " << beginVetoR
    << "\n\t endVetoR   = " << endVetoR
    << "\n\t diff1 = " << diff1
    << "\n\t cond1 = " << cond1
    << "\n\t diff2 = " << diff2
    << "\n\t cond2 = " << cond2
    << "\n\t diff3 = " << diff3
    << "\n\t cond3 = " << cond3
    << std::endl;

  // check if value is in range
  // for begin <= end takes [begin, end]
  // for begin >= end takes [begin, end] over zero angle!
  bool passWindow = false;
  if( cond1 && (cond2 && cond3 ) )      passWindow=true;
  else if( !cond1 && (cond2 || cond3) ) passWindow=true;
  else{
    return false;
  }

  if( passWindow ){
    if( beginVetoR==endVetoR && beginVetoR==(Type1)-1 ) return true;

    unsigned int diffV1 = endVetoR - beginVetoR;
    unsigned int diffV2 = bitNumber - beginVetoR;
    unsigned int diffV3 = endVetoR - bitNumber;

    bool condV1 = ( (diffV1>>nEtaBits) & 1 ) ? false : true;
    bool condV2 = ( (diffV2>>nEtaBits) & 1 ) ? false : true;
    bool condV3 = ( (diffV3>>nEtaBits) & 1 ) ? false : true;

    if( condV1 && !(condV2 && condV3) ) return true;
    else if( !condV1 && !(condV2 || condV3) ) return true;
    else{
      return false;
    }
  }
  else{
    LogDebug("l1t|Global") << "=====> ConditionEvaluation::checkRange: I should never be here." << std::endl;
    return false;
  }

  LogDebug("l1t|Global") << "=====> HELP!! I'm trapped and I cannot escape! AHHHHHH" << std::endl;

 }



/// check if a value is in a given range and outside of a veto range
template<class Type1> const bool ConditionEvaluation::checkRangePhi(const unsigned int bitNumber, 
									 const Type1& beginR, const Type1& endR, 
									 const Type1& beginVetoR, const Type1& endVetoR ) const {

  // set condtion to true if beginR==endR = default -1
  if( beginR==endR && beginR==(Type1)-1 ){
    return true;
  }

  int diff1 = endR - beginR;
  int diff2 = bitNumber - beginR;
  int diff3 = endR - bitNumber;

  bool cond1 = ( diff1<0 ) ? false : true;
  bool cond2 = ( diff2<0 ) ? false : true;
  bool cond3 = ( diff3<0 ) ? false : true;

  LogDebug("l1t|Global")
    << "\n l1t::ConditionEvaluation"
    << "\n\t bitNumber = " << bitNumber
    << "\n\t beginR = " << beginR
    << "\n\t endR   = " << endR
    << "\n\t beginVetoR = " << beginVetoR
    << "\n\t endVetoR   = " << endVetoR
    << "\n\t diff1 = " << diff1
    << "\n\t cond1 = " << cond1
    << "\n\t diff2 = " << diff2
    << "\n\t cond2 = " << cond2
    << "\n\t diff3 = " << diff3
    << "\n\t cond3 = " << cond3
    << std::endl;

  // check if value is in range
  // for begin <= end takes [begin, end]
  // for begin >= end takes [begin, end] over zero angle!
  bool passWindow = false;
  if( cond1 && (cond2 && cond3 ) )      passWindow=true;
  else if( !cond1 && (cond2 || cond3) ) passWindow=true;
  else{
    return false;
  }

  if( passWindow ){
    if( beginVetoR==endVetoR && beginVetoR==(Type1)-1 ) return true;

    int diffV1 = endVetoR - beginVetoR;
    int diffV2 = bitNumber - beginVetoR;
    int diffV3 = endVetoR - bitNumber;

    bool condV1 = ( diffV1<0 ) ? false : true;
    bool condV2 = ( diffV2<0 ) ? false : true;
    bool condV3 = ( diffV3<0 ) ? false : true;

    if( condV1 && !(condV2 && condV3) ) return true;
    else if( !condV1 && !(condV2 || condV3) ) return true;
    else{
      return false;
    }
  }
  else{
    LogDebug("l1t|Global") << "=====> ConditionEvaluation::checkRange: I should never be here." << std::endl;
    return false;
  }

  LogDebug("l1t|Global") << "=====> HELP!! I'm trapped and I cannot escape! AHHHHHH" << std::endl;

 }

 template<class Type1> const bool ConditionEvaluation::checkRangeDeltaEta(const unsigned int obj1Eta, const unsigned int obj2Eta, 
									  const Type1& lowerR, const Type1& upperR,
									  const unsigned int nEtaBits )  const {

/*   // set condtion to true if beginR==endR = default -1 */
/*   if( beginR==endR && beginR==-1 ){ */
/*     return true; */
/*   } */

  unsigned int compare = obj1Eta - obj2Eta;
  bool cond = ( (compare>>nEtaBits) & 1 ) ? false : true;

  unsigned int larger, smaller;
  if( cond ){
    larger = obj1Eta;
    smaller= obj2Eta;
  }
  else{
    larger = obj2Eta;
    smaller= obj1Eta;
  }

  unsigned int diff = ( ( larger + ((~smaller + 1) & 255) ) & 255);

  unsigned int diff1 = upperR - lowerR;
  unsigned int diff2 = diff - lowerR;
  unsigned int diff3 = upperR - diff;

  bool cond1 = ( (diff1>>nEtaBits) & 1 ) ? false : true;
  bool cond2 = ( (diff2>>nEtaBits) & 1 ) ? false : true;
  bool cond3 = ( (diff3>>nEtaBits) & 1 ) ? false : true;

  LogDebug("l1t|Global")
    << "\n l1t::ConditionEvaluation"
    << "\n\t obj1Eta = " << obj1Eta
    << "\n\t obj2Eta = " << obj2Eta
    << "\n\t lowerR = " << lowerR
    << "\n\t upperR = " << upperR
    << "\n\t compare = " << compare
    << "\n\t cond = " << cond
    << "\n\t diff = " << diff
    << "\n\t diff1 = " << diff1
    << "\n\t cond1 = " << cond1
    << "\n\t diff2 = " << diff2
    << "\n\t cond2 = " << cond2
    << "\n\t diff3 = " << diff3
    << "\n\t cond3 = " << cond3
    << std::endl;

  if( cond1 && (cond2 && cond3 ) )      return true;
  else if( !cond1 && (cond2 || cond3) ) return true;
  else{
    return false;
  }

 }



template<class Type1> const bool ConditionEvaluation::checkRangeDeltaPhi(const unsigned int obj1Phi, const unsigned int obj2Phi, 
									      const Type1& lowerR, const Type1& upperR )  const {

  int deltaPhi = abs(obj1Phi-obj2Phi);
  if( deltaPhi>71 ) deltaPhi = 143 - deltaPhi + 1; // Add +1 if the calculation is over 0

  int diff1 = upperR - lowerR;
  int diff2 = deltaPhi - lowerR;
  int diff3 = upperR - deltaPhi;

  bool cond1 = ( diff1<0 ) ? false : true;
  bool cond2 = ( diff2<0 ) ? false : true;
  bool cond3 = ( diff3<0 ) ? false : true;

  LogDebug("l1t|Global")
    << "\n l1t::ConditionEvaluation"
    << "\n\t obj1Phi = " << obj1Phi
    << "\n\t obj2Phi = " << obj2Phi
    << "\n\t deltaPhi = " << deltaPhi
    << "\n\t lowerR = " << lowerR
    << "\n\t upperR = " << upperR
    << "\n\t diff1 = " << diff1
    << "\n\t cond1 = " << cond1
    << "\n\t diff2 = " << diff2
    << "\n\t cond2 = " << cond2
    << "\n\t diff3 = " << diff3
    << "\n\t cond3 = " << cond3
    << std::endl;

  // check if value is in range
  // for begin <= end takes [begin, end]
  // for begin >= end takes [begin, end] over zero angle!
  if( cond1 && (cond2 && cond3 ) )      return true;
  else if( !cond1 && (cond2 || cond3) ) return true;
  else{
    return false;
  }

 }


}
#endif

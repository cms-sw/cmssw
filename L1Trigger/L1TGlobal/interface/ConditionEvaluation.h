#ifndef L1Trigger_L1TGlobal_ConditionEvaluation_h
#define L1Trigger_L1TGlobal_ConditionEvaluation_h

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
 *          Vladimir Rekovic - extend for indexing
 *
 *
 */

// system include files
#include <iostream>

#include <string>
#include <vector>

// user include files

//   base class

//
#include "DataFormats/L1TGlobal/interface/GlobalObjectMapFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <cstdint>

// forward declarations

namespace l1t {

  // class interface
  class ConditionEvaluation {
  public:
    /// constructor
    ConditionEvaluation() : m_condMaxNumberObjects(0), m_condLastResult(false), m_verbosity(0) {}

    /// destructor
    virtual ~ConditionEvaluation() {}

  public:
    /// get / set the maximum number of objects received for the
    /// evaluation of the condition
    inline int condMaxNumberObjects() const { return m_condMaxNumberObjects; }

    inline void setCondMaxNumberObjects(int condMaxNumberObjectsValue) {
      m_condMaxNumberObjects = condMaxNumberObjectsValue;
    }

    /// get the latest result for the condition
    inline bool condLastResult() const { return m_condLastResult; }

    /// call evaluateCondition and save last result
    inline void evaluateConditionStoreResult(const int bxEval) { m_condLastResult = evaluateCondition(bxEval); }

    /// the core function to check if the condition matches
    virtual const bool evaluateCondition(const int bxEval) const = 0;

    /// get numeric expression
    virtual std::string getNumericExpression() const {
      if (m_condLastResult) {
        return "1";
      } else {
        return "0";
      }
    }

    /// get all the object combinations evaluated to true in the condition
    inline CombinationsInCond const& getCombinationsInCond() const { return m_combinationsInCond; }

    /// print condition
    virtual void print(std::ostream& myCout) const;

    inline void setVerbosity(const int verbosity) { m_verbosity = verbosity; }

  protected:
    /// get all the object combinations (to fill it...)
    inline CombinationsInCond& combinationsInCond() const { return m_combinationsInCond; }

    /// check if a value is greater than a threshold or
    /// greater-or-equal depending on the value of the condGEqValue flag
    template <class Type1, class Type2>
    const bool checkThreshold(const Type1& thresholdL,
                              const Type1& thresholdH,
                              const Type2& value,
                              bool condGEqValue) const;

    /// check if a value is greater than a threshold or
    /// greater-or-equal depending on the value of the condGEqValue flag
    /// Added by Rick Cavanaugh for Displaced Muons:
    ///       Above checkThreshold fails when value overflows or threshold window is invalid
    ///       Below checkUnconstrainedPt allows value to overflow and only evaluates cut if threshold window is valid
    template <class Type1, class Type2>
    const bool checkUnconstrainedPt(const Type1& thresholdL,
                                    const Type1& thresholdH,
                                    const Type2& value,
                                    bool condGEqValue) const;

    /// check if a index is in a given range
    template <class Type1>
    const bool checkIndex(const Type1& indexLo, const Type1& indexHi, const unsigned int index) const;

    /// check if a bit with a given number is set in a mask
    template <class Type1>
    const bool checkBit(const Type1& mask, const unsigned int bitNumber) const;

    /// check if a value is in a given range and outside of a veto range
    template <class Type1>
    const bool checkRangeEta(const unsigned int bitNumber,
                             const Type1& W1beginR,
                             const Type1& W1endR,
                             const Type1& W2beginR,
                             const Type1& W2endR,
                             const unsigned int nEtaBits) const;

    /// check if a value is in a given range and outside of a veto range
    template <class Type1>
    const bool checkRangePhi(const unsigned int bitNumber,
                             const Type1& W1beginR,
                             const Type1& W1endR,
                             const Type1& W2beginR,
                             const Type1& W2endR) const;

    /// check if a value is in a given range
    template <class Type1>
    const bool checkRangeDeltaEta(const unsigned int obj1Eta,
                                  const unsigned int obj2Eta,
                                  const Type1& lowerR,
                                  const Type1& upperR,
                                  const unsigned int nEtaBits) const;

    /// check if a value is in a given range
    template <class Type1>
    const bool checkRangeDeltaPhi(const unsigned int obj1Phi,
                                  const unsigned int obj2Phi,
                                  const Type1& lowerR,
                                  const Type1& upperR) const;

    /// check if a value is in a given range
    template <class Type1>
    const bool checkRangeTfMuonIndex(const unsigned int bitNumber,
                                     const std::vector<Type1>& windows) const;

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
  template <class Type1, class Type2>
  const bool ConditionEvaluation::checkThreshold(const Type1& thresholdL,
                                                 const Type1& thresholdH,
                                                 const Type2& value,
                                                 const bool condGEqValue) const {
    if (value > 0) {
      LogTrace("L1GlobalTrigger") << "  checkThreshold check for condGEqValue = " << condGEqValue
                                  << "\n    hex: " << std::hex << "threshold = " << thresholdL << " - " << thresholdH
                                  << " value = " << value << "\n    dec: " << std::dec << "threshold = " << thresholdL
                                  << " - " << thresholdH << " value = " << value << std::endl;
    }

    if (condGEqValue) {
      if (value >= (Type2)thresholdL && (Type1)value < thresholdH) {
        //LogTrace("L1GlobalTrigger") << "    condGEqValue: value >= threshold"
        //    << std::endl;

        return true;
      }

      return false;

    } else {
      if (value == (Type2)thresholdL) {
        //LogTrace("L1GlobalTrigger") << "    condGEqValue: value = threshold"
        //    << std::endl;

        return true;
      }

      return false;
    }
  }

  // check if a value is greater than a threshold or
  // greater-or-equal depending on the value of the condGEqValue flag
  /// Added by Rick Cavanaugh for Displaced Muons:
  ///       Above checkThreshold fails when value overflows or threshold window is invalid
  ///       Below checkUnconstrainedPt allows value to overflow and only evaluates cut if threshold window is valid
  template <class Type1, class Type2>
  const bool ConditionEvaluation::checkUnconstrainedPt(const Type1& thresholdL,
                                                       const Type1& thresholdH,
                                                       const Type2& value,
                                                       const bool condGEqValue) const {
    if (value > 0) {
      LogTrace("L1GlobalTrigger") << "  checkUnconstrainedPt check for condGEqValue = " << condGEqValue
                                  << "\n    hex: " << std::hex << "threshold = " << thresholdL << " - " << thresholdH
                                  << " value = " << value << "\n    dec: " << std::dec << "threshold = " << thresholdL
                                  << " - " << thresholdH << " value = " << value << std::endl;
    }
    if (thresholdH > 0)  // Only evaluate cut if threshold window is valid
    {
      if (condGEqValue) {
        if (value >= (Type2)thresholdL && (Type1)value <= thresholdH) {
          return true;
        }
        return false;
      } else {
        if (value == (Type2)thresholdL) {
          return true;
        }
        return false;
      }
    } else  // If invalid threshold window, do not evaluate cut (ie. pass through)
      return true;
  }

  // check if a index in a given range
  template <class Type1>
  const bool ConditionEvaluation::checkIndex(const Type1& indexLo,
                                             const Type1& indexHi,
                                             const unsigned int index) const {
    LogDebug("l1t|Global") << "\n l1t::ConditionEvaluation"
                           << "\n\t indexLo = " << indexLo << "\n\t indexHi = " << indexHi << "\n\t index = " << index
                           << std::endl;

    // set condtion to false if indexLo > indexHi
    if (indexLo > indexHi) {
      return false;
    }
    if (index >= indexLo && index <= indexHi) {
      return true;
    }

    return false;
  }

  // check if a bit with a given number is set in a mask
  template <class Type1>
  const bool ConditionEvaluation::checkBit(const Type1& mask, const unsigned int bitNumber) const {
    uint64_t oneBit = 1ULL;

    if (bitNumber >= (sizeof(oneBit) * 8)) {
      if (m_verbosity) {
        LogTrace("L1GlobalTrigger") << "    checkBit "
                                    << "\n     Bit number = " << bitNumber << " larger than maximum allowed "
                                    << sizeof(oneBit) * 8 << std::endl;
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
  template <class Type1>
  const bool ConditionEvaluation::checkRangeEta(const unsigned int bitNumber,
                                                const Type1& W1beginR,
                                                const Type1& W1endR,
                                                const Type1& W2beginR,
                                                const Type1& W2endR,
                                                const unsigned int nEtaBits) const {
    // set condtion to true if beginR==endR = default -1
    if (W1beginR == W1endR && W1beginR == (Type1)-1) {
      return true;
    }

    unsigned int W1diff1 = W1endR - W1beginR;
    unsigned int W1diff2 = bitNumber - W1beginR;
    unsigned int W1diff3 = W1endR - bitNumber;

    bool W1cond1 = ((W1diff1 >> nEtaBits) & 1) ? false : true;
    bool W1cond2 = ((W1diff2 >> nEtaBits) & 1) ? false : true;
    bool W1cond3 = ((W1diff3 >> nEtaBits) & 1) ? false : true;

    // check if value is in range
    // for begin <= end takes [begin, end]
    // for begin >= end takes [begin, end] over zero angle!
    bool passWindow1 = false;
    if (W1cond1 && (W1cond2 && W1cond3))
      passWindow1 = true;
    else if (!W1cond1 && (W1cond2 || W1cond3))
      passWindow1 = true;
    else {
      passWindow1 = false;
    }

    LogDebug("l1t|Global") << "\n l1t::ConditionEvaluation"
                           << "\n\t bitNumber = " << bitNumber << "\n\t W1beginR = " << W1beginR
                           << "\n\t W1endR   = " << W1endR << "\n\t W1diff1 = " << W1diff1
                           << "\n\t W1cond1 = " << W1cond1 << "\n\t W1diff2 = " << W1diff2
                           << "\n\t W1cond2 = " << W1cond2 << "\n\t W1diff3 = " << W1diff3
                           << "\n\t W1cond3 = " << W1cond3 << "\n\t passWindow1 = " << passWindow1 << std::endl;

    if (W2beginR == W2endR && W2beginR == (Type1)-1) {
      return passWindow1;
    }

    unsigned int W2diff1 = W2endR - W2beginR;
    unsigned int W2diff2 = bitNumber - W2beginR;
    unsigned int W2diff3 = W2endR - bitNumber;

    bool W2cond1 = ((W2diff1 >> nEtaBits) & 1) ? false : true;
    bool W2cond2 = ((W2diff2 >> nEtaBits) & 1) ? false : true;
    bool W2cond3 = ((W2diff3 >> nEtaBits) & 1) ? false : true;

    bool passWindow2 = false;
    if (W2cond1 && (W2cond2 && W2cond3))
      passWindow2 = true;
    else if (!W2cond1 && (W2cond2 || W2cond3))
      passWindow2 = true;
    else {
      passWindow2 = false;
    }

    LogDebug("l1t|Global") << "\n\t W2beginR = " << W2beginR << "\n\t W2endR   = " << W2endR
                           << "\n\t W2diff1 = " << W2diff1 << "\n\t W2cond1 = " << W2cond1
                           << "\n\t W2diff2 = " << W2diff2 << "\n\t W2cond2 = " << W2cond2
                           << "\n\t W2diff3 = " << W2diff3 << "\n\t W2cond3 = " << W2cond3
                           << "\n\t passWindow2 = " << passWindow2
                           << "\n\t pass W1 || W2 = " << (passWindow1 || passWindow2) << std::endl;

    if (passWindow1 || passWindow2) {
      return true;
    } else {
      return false;
    }
  }

  /// check if a value is in a given range and outside of a veto range
  template <class Type1>
  const bool ConditionEvaluation::checkRangePhi(const unsigned int bitNumber,
                                                const Type1& W1beginR,
                                                const Type1& W1endR,
                                                const Type1& W2beginR,
                                                const Type1& W2endR) const {
    // set condtion to true if beginR==endR = default -1
    if (W1beginR == W1endR && W1beginR == (Type1)-1) {
      return true;
    }

    int W1diff1 = W1endR - W1beginR;
    int W1diff2 = bitNumber - W1beginR;
    int W1diff3 = W1endR - bitNumber;

    bool W1cond1 = (W1diff1 < 0) ? false : true;
    bool W1cond2 = (W1diff2 < 0) ? false : true;
    bool W1cond3 = (W1diff3 < 0) ? false : true;

    // check if value is in range
    // for begin <= end takes [begin, end]
    // for begin >= end takes [begin, end] over zero angle!
    bool passWindow1 = false;
    if (W1cond1 && (W1cond2 && W1cond3))
      passWindow1 = true;
    else if (!W1cond1 && (W1cond2 || W1cond3))
      passWindow1 = true;
    else {
      passWindow1 = false;
    }

    LogDebug("l1t|Global") << "\n l1t::ConditionEvaluation"
                           << "\n\t bitNumber = " << bitNumber << "\n\t W1beginR = " << W1beginR
                           << "\n\t W1endR   = " << W1endR << "\n\t W1diff1 = " << W1diff1
                           << "\n\t W1cond1 = " << W1cond1 << "\n\t W1diff2 = " << W1diff2
                           << "\n\t W1cond2 = " << W1cond2 << "\n\t W1diff3 = " << W1diff3
                           << "\n\t W1cond3 = " << W1cond3 << std::endl;

    if (W2beginR == W2endR && W2beginR == (Type1)-1) {
      return passWindow1;
    }

    int W2diff1 = W2endR - W2beginR;
    int W2diff2 = bitNumber - W2beginR;
    int W2diff3 = W2endR - bitNumber;

    bool W2cond1 = (W2diff1 < 0) ? false : true;
    bool W2cond2 = (W2diff2 < 0) ? false : true;
    bool W2cond3 = (W2diff3 < 0) ? false : true;

    // check if value is in range
    // for begin <= end takes [begin, end]
    // for begin >= end takes [begin, end] over zero angle!
    bool passWindow2 = false;
    if (W2cond1 && (W2cond2 && W2cond3))
      passWindow2 = true;
    else if (!W2cond1 && (W2cond2 || W2cond3))
      passWindow2 = true;
    else {
      passWindow2 = false;
    }

    if (passWindow1 || passWindow2) {
      return true;
    } else {
      return false;
    }
  }

  template <class Type1>
  const bool ConditionEvaluation::checkRangeDeltaEta(const unsigned int obj1Eta,
                                                     const unsigned int obj2Eta,
                                                     const Type1& lowerR,
                                                     const Type1& upperR,
                                                     const unsigned int nEtaBits) const {
    /*   // set condtion to true if beginR==endR = default -1 */
    /*   if( beginR==endR && beginR==-1 ){ */
    /*     return true; */
    /*   } */

    unsigned int compare = obj1Eta - obj2Eta;
    bool cond = ((compare >> nEtaBits) & 1) ? false : true;

    unsigned int larger, smaller;
    if (cond) {
      larger = obj1Eta;
      smaller = obj2Eta;
    } else {
      larger = obj2Eta;
      smaller = obj1Eta;
    }

    unsigned int diff = ((larger + ((~smaller + 1) & 255)) & 255);

    unsigned int diff1 = upperR - lowerR;
    unsigned int diff2 = diff - lowerR;
    unsigned int diff3 = upperR - diff;

    bool cond1 = ((diff1 >> nEtaBits) & 1) ? false : true;
    bool cond2 = ((diff2 >> nEtaBits) & 1) ? false : true;
    bool cond3 = ((diff3 >> nEtaBits) & 1) ? false : true;

    LogDebug("l1t|Global") << "\n l1t::ConditionEvaluation"
                           << "\n\t obj1Eta = " << obj1Eta << "\n\t obj2Eta = " << obj2Eta << "\n\t lowerR = " << lowerR
                           << "\n\t upperR = " << upperR << "\n\t compare = " << compare << "\n\t cond = " << cond
                           << "\n\t diff = " << diff << "\n\t diff1 = " << diff1 << "\n\t cond1 = " << cond1
                           << "\n\t diff2 = " << diff2 << "\n\t cond2 = " << cond2 << "\n\t diff3 = " << diff3
                           << "\n\t cond3 = " << cond3 << std::endl;

    if (cond1 && (cond2 && cond3))
      return true;
    else if (!cond1 && (cond2 || cond3))
      return true;
    else {
      return false;
    }
  }

  template <class Type1>
  const bool ConditionEvaluation::checkRangeDeltaPhi(const unsigned int obj1Phi,
                                                     const unsigned int obj2Phi,
                                                     const Type1& lowerR,
                                                     const Type1& upperR) const {
    int deltaPhi = abs(int(obj1Phi) - int(obj2Phi));
    if (deltaPhi > 71)
      deltaPhi = 143 - deltaPhi + 1;  // Add +1 if the calculation is over 0

    int diff1 = upperR - lowerR;
    int diff2 = deltaPhi - lowerR;
    int diff3 = upperR - deltaPhi;

    bool cond1 = (diff1 < 0) ? false : true;
    bool cond2 = (diff2 < 0) ? false : true;
    bool cond3 = (diff3 < 0) ? false : true;

    LogDebug("l1t|Global") << "\n l1t::ConditionEvaluation"
                           << "\n\t obj1Phi = " << obj1Phi << "\n\t obj2Phi = " << obj2Phi
                           << "\n\t deltaPhi = " << deltaPhi << "\n\t lowerR = " << lowerR << "\n\t upperR = " << upperR
                           << "\n\t diff1 = " << diff1 << "\n\t cond1 = " << cond1 << "\n\t diff2 = " << diff2
                           << "\n\t cond2 = " << cond2 << "\n\t diff3 = " << diff3 << "\n\t cond3 = " << cond3
                           << std::endl;

    // check if value is in range
    // for begin <= end takes [begin, end]
    // for begin >= end takes [begin, end] over zero angle!
    if (cond1 && (cond2 && cond3))
      return true;
    else if (!cond1 && (cond2 || cond3))
      return true;
    else {
      return false;
    }
  }

  template <class Type1>
  const bool ConditionEvaluation::checkRangeTfMuonIndex(const unsigned int value,
                                                        const std::vector<Type1>& windows) const {
    if (windows.empty()) {
      return true;
    }

    for (const auto& window : windows) {
      if ((window.lower <= value) and (value <= window.upper)) {
        return true;
      }
    }

    return false;
  }

}  // namespace l1t
#endif

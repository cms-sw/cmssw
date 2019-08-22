/**
 * \class L1GtTriggerMenuLite
 *
 *
 * Description: L1 trigger menu and masks, lite version not using event setup.
 *
 * Implementation:
 *    This is the lite version of the L1 trigger menu, with trigger masks included,
 *    to be used in the environments not having access to event setup. It offers
 *    limited access to the full L1 trigger menu which is implemented as event setup
 *    (CondFormats/L1TObjects/interface/L1GtTriggerMenu.h). The masks are provided for
 *    the physics partition only.
 *
 *    An EDM product is created and saved in the Run Data, under the assumption that the
 *    menu remains the same in a run.
 *    The corresponding producer will read the full L1 trigger menu and the trigger masks
 *    from event setup, fill the corresponding members and save it as EDM product.
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
 *
 */

// this class header
#include "DataFormats/L1GlobalTrigger/interface/L1GtTriggerMenuLite.h"

// system include files
#include <iostream>
#include <iomanip>

// user include files

// forward declarations

// constructor
L1GtTriggerMenuLite::L1GtTriggerMenuLite()
    : m_triggerMenuInterface("NULL"),
      m_triggerMenuName("NULL"),
      m_triggerMenuImplementation("NULL"),
      m_scaleDbKey("NULL") {
  // empty
}

L1GtTriggerMenuLite::L1GtTriggerMenuLite(const std::string& triggerMenuNameVal,
                                         const L1TriggerMap& algorithmMapVal,
                                         const L1TriggerMap& algorithmAliasMapVal,
                                         const L1TriggerMap& technicalTriggerMapVal,
                                         const std::vector<unsigned int>& triggerMaskAlgoTrigVal,
                                         const std::vector<unsigned int>& triggerMaskTechTrigVal,
                                         const std::vector<std::vector<int> >& prescaleFactorsAlgoTrigVal,
                                         const std::vector<std::vector<int> >& prescaleFactorsTechTrigVal)
    : m_triggerMenuInterface("NULL"),
      m_triggerMenuName(triggerMenuNameVal),
      m_triggerMenuImplementation("NULL"),
      m_scaleDbKey("NULL"),
      m_algorithmMap(algorithmMapVal),
      m_algorithmAliasMap(algorithmAliasMapVal),
      m_technicalTriggerMap(technicalTriggerMapVal),
      m_triggerMaskAlgoTrig(triggerMaskAlgoTrigVal),
      m_triggerMaskTechTrig(triggerMaskTechTrigVal),
      m_prescaleFactorsAlgoTrig(prescaleFactorsAlgoTrigVal),
      m_prescaleFactorsTechTrig(prescaleFactorsTechTrigVal)

{
  // empty
}

// copy constructor
L1GtTriggerMenuLite::L1GtTriggerMenuLite(const L1GtTriggerMenuLite& rhs) {
  m_triggerMenuInterface = rhs.m_triggerMenuInterface;
  m_triggerMenuName = rhs.m_triggerMenuName;
  m_triggerMenuImplementation = rhs.m_triggerMenuImplementation;
  m_scaleDbKey = rhs.m_scaleDbKey;

  // copy algorithm map
  m_algorithmMap = rhs.m_algorithmMap;
  m_algorithmAliasMap = rhs.m_algorithmAliasMap;

  // copy technical triggers
  // (separate map for technical triggers and physics triggers)
  m_technicalTriggerMap = rhs.m_technicalTriggerMap;

  // copy masks
  m_triggerMaskAlgoTrig = rhs.m_triggerMaskAlgoTrig;
  m_triggerMaskTechTrig = rhs.m_triggerMaskTechTrig;

  // copy prescale factors
  m_prescaleFactorsAlgoTrig = rhs.m_prescaleFactorsAlgoTrig;
  m_prescaleFactorsTechTrig = rhs.m_prescaleFactorsTechTrig;
}

// destructor
L1GtTriggerMenuLite::~L1GtTriggerMenuLite() {
  m_algorithmMap.clear();
  m_algorithmAliasMap.clear();
  m_technicalTriggerMap.clear();
}

// assignment operator
L1GtTriggerMenuLite& L1GtTriggerMenuLite::operator=(const L1GtTriggerMenuLite& rhs) {
  if (this != &rhs) {
    m_triggerMenuInterface = rhs.m_triggerMenuInterface;
    m_triggerMenuName = rhs.m_triggerMenuName;
    m_triggerMenuImplementation = rhs.m_triggerMenuImplementation;
    m_scaleDbKey = rhs.m_scaleDbKey;

    m_algorithmMap = rhs.m_algorithmMap;
    m_algorithmAliasMap = rhs.m_algorithmAliasMap;

    m_technicalTriggerMap = rhs.m_technicalTriggerMap;

    m_triggerMaskAlgoTrig = rhs.m_triggerMaskAlgoTrig;
    m_triggerMaskTechTrig = rhs.m_triggerMaskTechTrig;

    m_prescaleFactorsAlgoTrig = rhs.m_prescaleFactorsAlgoTrig;
    m_prescaleFactorsTechTrig = rhs.m_prescaleFactorsTechTrig;
  }

  // return the object
  return *this;
}

// equal operator
bool L1GtTriggerMenuLite::operator==(const L1GtTriggerMenuLite& rhs) const {
  if (m_triggerMenuInterface != rhs.m_triggerMenuInterface) {
    return false;
  }

  if (m_triggerMenuName != rhs.m_triggerMenuName) {
    return false;
  }

  if (m_triggerMenuImplementation != rhs.m_triggerMenuImplementation) {
    return false;
  }

  if (m_scaleDbKey != rhs.m_scaleDbKey) {
    return false;
  }

  if (m_algorithmMap != rhs.m_algorithmMap) {
    return false;
  }

  if (m_algorithmAliasMap != rhs.m_algorithmAliasMap) {
    return false;
  }

  if (m_technicalTriggerMap != rhs.m_technicalTriggerMap) {
    return false;
  }

  if (m_triggerMaskAlgoTrig != rhs.m_triggerMaskAlgoTrig) {
    return false;
  }

  if (m_triggerMaskTechTrig != rhs.m_triggerMaskTechTrig) {
    return false;
  }

  if (m_prescaleFactorsAlgoTrig != rhs.m_prescaleFactorsAlgoTrig) {
    return false;
  }

  if (m_prescaleFactorsTechTrig != rhs.m_prescaleFactorsTechTrig) {
    return false;
  }

  // all members identical
  return true;
}

// unequal operator
bool L1GtTriggerMenuLite::operator!=(const L1GtTriggerMenuLite& otherObj) const { return !(otherObj == *this); }

// merge rule: test on isProductEqual
bool L1GtTriggerMenuLite::isProductEqual(const L1GtTriggerMenuLite& otherObj) const { return (otherObj == *this); }

// set the trigger menu name
void L1GtTriggerMenuLite::setGtTriggerMenuInterface(const std::string& menuInterface) {
  m_triggerMenuInterface = menuInterface;
}

void L1GtTriggerMenuLite::setGtTriggerMenuName(const std::string& menuName) { m_triggerMenuName = menuName; }

void L1GtTriggerMenuLite::setGtTriggerMenuImplementation(const std::string& menuImplementation) {
  m_triggerMenuImplementation = menuImplementation;
}

// set menu associated scale key
void L1GtTriggerMenuLite::setGtScaleDbKey(const std::string& scaleKey) { m_scaleDbKey = scaleKey; }

// set the algorithm map (by algorithm names)
void L1GtTriggerMenuLite::setGtAlgorithmMap(const L1TriggerMap& algoMap) { m_algorithmMap = algoMap; }

// set the algorithm map (by algorithm aliases)
void L1GtTriggerMenuLite::setGtAlgorithmAliasMap(const L1TriggerMap& algoMap) { m_algorithmAliasMap = algoMap; }

// set the technical trigger map
void L1GtTriggerMenuLite::setGtTechnicalTriggerMap(const L1TriggerMap& ttMap) { m_technicalTriggerMap = ttMap; }

// set the trigger mask for physics algorithms
void L1GtTriggerMenuLite::setGtTriggerMaskAlgoTrig(const std::vector<unsigned int>& maskValue) {
  m_triggerMaskAlgoTrig = maskValue;
}

// set the trigger mask for technical triggers
void L1GtTriggerMenuLite::setGtTriggerMaskTechTrig(const std::vector<unsigned int>& maskValue) {
  m_triggerMaskTechTrig = maskValue;
}

// set the prescale factors
void L1GtTriggerMenuLite::setGtPrescaleFactorsAlgoTrig(const std::vector<std::vector<int> >& factorValue) {
  m_prescaleFactorsAlgoTrig = factorValue;
}

void L1GtTriggerMenuLite::setGtPrescaleFactorsTechTrig(const std::vector<std::vector<int> >& factorValue) {
  m_prescaleFactorsTechTrig = factorValue;
}

// print the trigger menu
void L1GtTriggerMenuLite::print(std::ostream& myCout, int& printVerbosity) const {
  //

  switch (printVerbosity) {
    case 0: {
      size_t nrDefinedAlgo = m_algorithmMap.size();
      size_t nrDefinedTech = m_technicalTriggerMap.size();

      // header for printing algorithms

      myCout << "\n   ********** L1 Trigger Menu - printing   ********** \n"
             << "\nL1 Trigger Menu Interface: " << m_triggerMenuInterface
             << "\nL1 Trigger Menu Name:      " << m_triggerMenuName
             << "\nL1 Trigger Menu Implementation: " << m_triggerMenuImplementation
             << "\nAssociated Scale DB Key: " << m_scaleDbKey << "\n\n"
             << "\nL1 Physics Algorithms: " << nrDefinedAlgo << " algorithms defined."
             << "\n\n"
             << "Bit Number " << std::right << std::setw(35) << "Algorithm Name"
             << "  " << std::right << std::setw(35) << "Algorithm Alias"
             << "  " << std::right << std::setw(12) << "Trigger Mask";
      for (unsigned iSet = 0; iSet < m_prescaleFactorsAlgoTrig.size(); iSet++) {
        myCout << std::right << std::setw(10) << "PF Set " << std::right << std::setw(2) << iSet;
      }

      myCout << std::endl;

      for (CItL1Trig itTrig = m_algorithmMap.begin(); itTrig != m_algorithmMap.end(); itTrig++) {
        const unsigned int bitNumber = itTrig->first;
        const std::string& aName = itTrig->second;

        std::string aAlias;
        CItL1Trig itAlias = m_algorithmAliasMap.find(bitNumber);
        if (itAlias != m_algorithmAliasMap.end()) {
          aAlias = itAlias->second;
        }

        myCout << std::setw(6) << bitNumber << "     " << std::right << std::setw(35) << aName << "  " << std::right
               << std::setw(35) << aAlias << "  " << std::right << std::setw(12) << m_triggerMaskAlgoTrig[bitNumber];
        for (unsigned iSet = 0; iSet < m_prescaleFactorsAlgoTrig.size(); iSet++) {
          myCout << std::right << std::setw(12) << m_prescaleFactorsAlgoTrig[iSet][bitNumber];
        }

        myCout << std::endl;
      }

      myCout << "\nL1 Technical Triggers: " << nrDefinedTech << " technical triggers defined."
             << "\n\n"
             << std::endl;
      if (nrDefinedTech) {
        myCout << std::right << std::setw(6) << "Bit Number " << std::right << std::setw(45)
               << " Technical trigger name "
               << "  " << std::right << std::setw(12) << "Trigger Mask";
        for (unsigned iSet = 0; iSet < m_prescaleFactorsTechTrig.size(); iSet++) {
          myCout << std::right << std::setw(10) << "PF Set " << std::right << std::setw(2) << iSet;
        }

        myCout << std::endl;
      }

      for (CItL1Trig itTrig = m_technicalTriggerMap.begin(); itTrig != m_technicalTriggerMap.end(); itTrig++) {
        unsigned int bitNumber = itTrig->first;
        std::string aName = itTrig->second;

        myCout << std::setw(6) << bitNumber << "       " << std::right << std::setw(45) << aName << std::right
               << std::setw(12) << m_triggerMaskTechTrig[bitNumber];
        for (unsigned iSet = 0; iSet < m_prescaleFactorsTechTrig.size(); iSet++) {
          myCout << std::right << std::setw(12) << m_prescaleFactorsTechTrig[iSet][bitNumber];
        }

        myCout << std::endl;
      }

    } break;
    default: {
      myCout << "\n   ********** L1 Trigger Menu - printing   ********** \n\n"
             << "Verbosity level: " << printVerbosity << " not implemented.\n\n"
             << std::endl;
    } break;
  }
}

// output stream operator
std::ostream& operator<<(std::ostream& streamRec, const L1GtTriggerMenuLite& result) {
  int verbosityLevel = 0;

  result.print(streamRec, verbosityLevel);
  return streamRec;
}

// get the alias for a physics algorithm with a given bit number
const std::string* L1GtTriggerMenuLite::gtAlgorithmAlias(const unsigned int bitNumber, int& errorCode) const {
  const std::string* gtAlgorithmAlias = nullptr;

  for (CItL1Trig itTrig = m_algorithmAliasMap.begin(); itTrig != m_algorithmAliasMap.end(); itTrig++) {
    if (itTrig->first == bitNumber) {
      gtAlgorithmAlias = &(itTrig->second);

      errorCode = 0;
      return gtAlgorithmAlias;
    }
  }

  errorCode = 1;
  return gtAlgorithmAlias;
}

// get the name for a physics algorithm or a technical trigger
// with a given bit number
const std::string* L1GtTriggerMenuLite::gtAlgorithmName(const unsigned int bitNumber, int& errorCode) const {
  const std::string* gtAlgorithmName = nullptr;

  for (CItL1Trig itTrig = m_algorithmMap.begin(); itTrig != m_algorithmMap.end(); itTrig++) {
    if (itTrig->first == bitNumber) {
      gtAlgorithmName = &(itTrig->second);

      errorCode = 0;
      return gtAlgorithmName;
    }
  }

  errorCode = 1;
  return gtAlgorithmName;
}

const std::string* L1GtTriggerMenuLite::gtTechTrigName(const unsigned int bitNumber, int& errorCode) const {
  const std::string* gtTechTrigName = nullptr;

  for (CItL1Trig itTrig = m_technicalTriggerMap.begin(); itTrig != m_technicalTriggerMap.end(); itTrig++) {
    if (itTrig->first == bitNumber) {
      gtTechTrigName = &(itTrig->second);

      errorCode = 0;
      return gtTechTrigName;
    }
  }

  errorCode = 1;
  return gtTechTrigName;
}

// get the bit number for a physics algorithm or a technical trigger
// with a given name or alias
const unsigned int L1GtTriggerMenuLite::gtBitNumber(const std::string& trigName, int& errorCode) const {
  unsigned int bitNr = 999;

  //
  for (CItL1Trig itTrig = m_algorithmAliasMap.begin(); itTrig != m_algorithmAliasMap.end(); itTrig++) {
    if (itTrig->second == trigName) {
      bitNr = itTrig->first;

      errorCode = 0;
      return bitNr;
    }
  }

  //
  for (CItL1Trig itTrig = m_algorithmMap.begin(); itTrig != m_algorithmMap.end(); itTrig++) {
    if (itTrig->second == trigName) {
      bitNr = itTrig->first;
      errorCode = 0;
      return bitNr;
    }
  }

  //
  for (CItL1Trig itTrig = m_technicalTriggerMap.begin(); itTrig != m_technicalTriggerMap.end(); itTrig++) {
    if (itTrig->second == trigName) {
      bitNr = itTrig->first;
      errorCode = 0;
      return bitNr;
    }
  }

  errorCode = 1;
  return bitNr;
}

// get the result for a physics algorithm or a technical trigger with name trigName
// use directly the format of decisionWord (no typedef)
const bool L1GtTriggerMenuLite::gtTriggerResult(const std::string& trigName,
                                                const std::vector<bool>& decWord,
                                                int& errorCode) const {
  bool trigResult = false;

  // try first physics algorithm aliases

  for (CItL1Trig itTrig = m_algorithmAliasMap.begin(); itTrig != m_algorithmAliasMap.end(); itTrig++) {
    if (itTrig->second == trigName) {
      unsigned int bitNumber = itTrig->first;

      if ((bitNumber >= decWord.size())) {
        trigResult = false;
        errorCode = 10;
      } else {
        trigResult = decWord[bitNumber];
        errorCode = 0;
      }

      return trigResult;
    }
  }

  // ... then physics algorithm names

  for (CItL1Trig itTrig = m_algorithmMap.begin(); itTrig != m_algorithmMap.end(); itTrig++) {
    if (itTrig->second == trigName) {
      unsigned int bitNumber = itTrig->first;

      if ((bitNumber >= decWord.size())) {
        trigResult = false;
        errorCode = 10;
      } else {
        trigResult = decWord[bitNumber];
        errorCode = 0;
      }

      return trigResult;
    }
  }

  // ... then technical trigger names

  for (CItL1Trig itTrig = m_technicalTriggerMap.begin(); itTrig != m_technicalTriggerMap.end(); itTrig++) {
    if (itTrig->second == trigName) {
      unsigned int bitNumber = itTrig->first;

      if ((bitNumber >= decWord.size())) {
        trigResult = false;
        errorCode = 10;
      } else {
        trigResult = decWord[bitNumber];
        errorCode = 0;
      }

      return trigResult;
    }
  }

  // algorithm or technical trigger not in the menu

  errorCode = 1;
  return false;
}

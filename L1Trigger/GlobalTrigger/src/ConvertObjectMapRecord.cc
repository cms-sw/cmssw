/**
 * \class ConvertObjectMapRecord
 *
 *
 * Description: see header file.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * \author: W. David Dagenhart
 *
 * $Date: 2012/03/02 22:03:26 $
 * $Revision: 1.1 $
 *
 */

#include "L1Trigger/GlobalTrigger/interface/ConvertObjectMapRecord.h"

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMaps.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtLogicParser.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/Exception.h"

ConvertObjectMapRecord::ConvertObjectMapRecord(const edm::ParameterSet& pset) :
  m_l1GtObjectMapTag(pset.getParameter<edm::InputTag>("L1GtObjectMapTag")) {

  produces<L1GlobalTriggerObjectMaps>();
}

ConvertObjectMapRecord::~ConvertObjectMapRecord() {
}

void ConvertObjectMapRecord::
produce(edm::Event& event, const edm::EventSetup& es) {

  // Read in the existing object from the data
  edm::Handle<L1GlobalTriggerObjectMapRecord> gtObjectMapRecord;
  event.getByLabel(m_l1GtObjectMapTag, gtObjectMapRecord);

  if (!gtObjectMapRecord.isValid()) {
    return;
  }

  // Create the new object we are going to copy the information to
  std::auto_ptr<L1GlobalTriggerObjectMaps> gtObjectMaps(new L1GlobalTriggerObjectMaps);

  // get the algorithm bit numbers and sort them
  std::vector<int> algoBitNumbers;
  std::vector<L1GlobalTriggerObjectMap> const& vectorInRecord = gtObjectMapRecord->gtObjectMap();
  algoBitNumbers.reserve(vectorInRecord.size());
  for (std::vector<L1GlobalTriggerObjectMap>::const_iterator i = vectorInRecord.begin(),
                                                          iEnd = vectorInRecord.end();
       i != iEnd; ++i) {
    algoBitNumbers.push_back(i->algoBitNumber());
  }
  edm::sort_all(algoBitNumbers);

  gtObjectMaps->reserveForAlgorithms(algoBitNumbers.size());

  if (!algoBitNumbers.empty() && algoBitNumbers[0] < 0) {
    cms::Exception ex("L1GlobalTrigger");
    ex << "Negative algorithm bit number";
    ex.addContext("Calling ConvertObjectMapRecord::produce");
    throw ex;
  }

  unsigned sizeOfNamesVector = 0;
  if (!algoBitNumbers.empty()) {
    sizeOfNamesVector = static_cast<unsigned>(algoBitNumbers.back()) + 1;
  }
  std::vector<std::string> savedNames(sizeOfNamesVector);

  // Loop over the object map record and copy the algorithm information
  // Just count the condition and index information so we can reserve
  // memory for them before filling them.
  unsigned startIndexOfConditions = 0;
  unsigned nIndexes = 0;

  for (std::vector<int>::const_iterator iBit = algoBitNumbers.begin(), endBits = algoBitNumbers.end();
       iBit != endBits; ++iBit) {
    L1GlobalTriggerObjectMap const* objMap = gtObjectMapRecord->getObjectMap(*iBit);

    gtObjectMaps->pushBackAlgorithm(startIndexOfConditions,
                                    objMap->algoBitNumber(),
                                    objMap->algoGtlResult());

    savedNames.at(static_cast<unsigned>(*iBit)) = objMap->algoName();

    std::vector<L1GtLogicParser::OperandToken> const& operandTokens =
      objMap->operandTokenVector();

    startIndexOfConditions += operandTokens.size();

    int tokenCounter = 0;
    for (std::vector<L1GtLogicParser::OperandToken>::const_iterator iToken = operandTokens.begin(),
                                                                 endTokens = operandTokens.end();
         iToken != endTokens;
         ++iToken, ++tokenCounter) {

      if (tokenCounter != iToken->tokenNumber) {
        cms::Exception ex("L1GlobalTrigger");
        ex << "Token numbers not sequential";
        ex.addContext("Calling ConvertObjectMapRecord::produce");
        throw ex;
      }

      CombinationsInCond const* combos = objMap->getCombinationsInCond(iToken->tokenNumber);
      for (CombinationsInCond::const_iterator iCombo = combos->begin(),
                                           endCombos = combos->end();
           iCombo != endCombos; ++iCombo) {
        for (std::vector<int>::const_iterator iIndex = iCombo->begin(),
                                          endIndexes = iCombo->end();
             iIndex != endIndexes; ++iIndex) {
          ++nIndexes;
        }
      }
    }
  }
  gtObjectMaps->reserveForConditions(startIndexOfConditions);
  gtObjectMaps->reserveForObjectIndexes(nIndexes);

  edm::ParameterSet namesPset;
  namesPset.addParameter<std::vector<std::string> >(std::string("@algorithmNames"), savedNames);

  // Now loop a second time and fill the condition and index
  // information.
  unsigned startIndexOfCombinations = 0;
  for (std::vector<int>::const_iterator iBit = algoBitNumbers.begin(), endBits = algoBitNumbers.end();
       iBit != endBits; ++iBit) {
     L1GlobalTriggerObjectMap const* objMap = gtObjectMapRecord->getObjectMap(*iBit);

     std::vector<L1GtLogicParser::OperandToken> const& operandTokens =
       objMap->operandTokenVector();

     savedNames.clear();
     if (savedNames.capacity() < operandTokens.size()) {
       savedNames.reserve(operandTokens.size());
     }

     for (std::vector<L1GtLogicParser::OperandToken>::const_iterator iToken = operandTokens.begin(),
                                                                  endTokens = operandTokens.end();
          iToken != endTokens; ++iToken) {

       savedNames.push_back(iToken->tokenName);

       unsigned short nObjectsPerCombination = 0;
       bool first = true;
       unsigned nIndexesInCombination = 0;

       CombinationsInCond const* combos = objMap->getCombinationsInCond(iToken->tokenNumber);
       for (CombinationsInCond::const_iterator iCombo = combos->begin(),
                                            endCombos = combos->end();
            iCombo != endCombos; ++iCombo) {
         if (first) {
           if (iCombo->size() > std::numeric_limits<unsigned short>::max()) {
             cms::Exception ex("L1GlobalTrigger");
             ex << "Number of objects per combination out of range";
             ex.addContext("Calling ConvertObjectMapRecord::produce");
             throw ex;
           }
           nObjectsPerCombination = iCombo->size();
           first = false;
         } else {
           if (nObjectsPerCombination != iCombo->size()) {
             cms::Exception ex("L1GlobalTrigger");
             ex << "inconsistent number of objects per condition";
             ex.addContext("Calling ConvertObjectMapRecord::produce");
             throw ex;
           }
         }

         for (std::vector<int>::const_iterator iIndex = iCombo->begin(),
                                           endIndexes = iCombo->end();
              iIndex != endIndexes; ++iIndex) {

           if (*iIndex < 0 || *iIndex > std::numeric_limits<unsigned char>::max()) {
             cms::Exception ex("L1GlobalTrigger");
             ex << "object index too large, out of range";
             ex.addContext("Calling ConvertObjectMapRecord::produce");
             throw ex;
           }
           gtObjectMaps->pushBackObjectIndex(*iIndex);
           ++nIndexesInCombination;
         }
       }
       gtObjectMaps->pushBackCondition(startIndexOfCombinations,
                                       nObjectsPerCombination,
                                       iToken->tokenResult);
       startIndexOfCombinations += nIndexesInCombination;
     }
     namesPset.addParameter<std::vector<std::string> >(objMap->algoName(), savedNames);
  }
  namesPset.registerIt();
  gtObjectMaps->setNamesParameterSetID(namesPset.id());

  gtObjectMaps->consistencyCheck();
  event.put(gtObjectMaps);
}

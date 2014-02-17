/**
 * \class CompareToObjectMapRecord
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

#include "L1Trigger/GlobalTrigger/interface/CompareToObjectMapRecord.h"

#include <algorithm>
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
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/Exception.h"


CompareToObjectMapRecord::CompareToObjectMapRecord(const edm::ParameterSet& pset) :
  m_l1GtObjectMapTag(pset.getParameter<edm::InputTag>("L1GtObjectMapTag")),
  m_l1GtObjectMapsTag(pset.getParameter<edm::InputTag>("L1GtObjectMapsTag")),
  verbose_(pset.getUntrackedParameter<bool>("verbose", false))
{
}

CompareToObjectMapRecord::~CompareToObjectMapRecord() {
}

void CompareToObjectMapRecord::
analyze(edm::Event const& event, edm::EventSetup const& es) {

  // Read in the data in the old format
  edm::Handle<L1GlobalTriggerObjectMapRecord> gtObjectMapRecord;
  event.getByLabel(m_l1GtObjectMapTag, gtObjectMapRecord);

  // Read in the data in the new format
  edm::Handle<L1GlobalTriggerObjectMaps> gtObjectMaps;
  event.getByLabel(m_l1GtObjectMapsTag, gtObjectMaps);

  // In the new format the names are not in the event data,
  // They are in the ParameterSet registry
  edm::pset::Registry* psetRegistry = edm::pset::Registry::instance();
  edm::ParameterSet const* pset = psetRegistry->getMapped(gtObjectMaps->namesParameterSetID());
  if (pset == 0) {
    cms::Exception ex("L1GlobalTrigger");
    ex << "Could not find L1 trigger names ParameterSet in the registry";
    ex.addContext("Calling CompareToObjectMapRecord::analyze");
    throw ex;
  }

  // First compare the algorithm bit numbers
  std::vector<int> algoBitNumbers1;
  std::vector<L1GlobalTriggerObjectMap> const& vectorInRecord = gtObjectMapRecord->gtObjectMap();
  algoBitNumbers1.reserve(vectorInRecord.size());
  for (std::vector<L1GlobalTriggerObjectMap>::const_iterator i = vectorInRecord.begin(),
                                                          iEnd = vectorInRecord.end();
       i != iEnd; ++i) {
    algoBitNumbers1.push_back(i->algoBitNumber());
    if (verbose_) {
      // This will print out all the data from the L1GlobalTriggerObjectMapRecord
      i->print(std::cout);
    }
  }
  edm::sort_all(algoBitNumbers1);

  std::vector<int> algoBitNumbers2;
  gtObjectMaps->getAlgorithmBitNumbers(algoBitNumbers2);
  if (algoBitNumbers1 != algoBitNumbers2) {
    cms::Exception ex("L1GlobalTrigger");
    ex << "Algorithm bit numbers do not match";
    ex.addContext("Calling CompareToObjectMapRecord::analyze");
    throw ex;
  }

  // Now test the algorithm names
  std::vector<std::string> algoNames2 = pset->getParameter<std::vector<std::string> >("@algorithmNames");

  // In the ParameterSet, the algorithm names are referenced by position
  // in the vector. If the bit number for a position in the vector is
  // not assigned, then the algorithm name should be an empty string.
  for (int i = 0; i < static_cast<int>(algoNames2.size()); ++i) {
    if (!std::binary_search(algoBitNumbers1.begin(), algoBitNumbers1.end(), i)) {
      if (algoNames2[i] != std::string("")) {
        cms::Exception ex("L1GlobalTrigger");
        ex << "Undefined algorithm should have empty name";
        ex.addContext("Calling CompareToObjectMapRecord::analyze");
        throw ex;
      }
    }
  }

  // Main loop over algorithms
  for (std::vector<int>::const_iterator iBit = algoBitNumbers1.begin(), endBits = algoBitNumbers1.end();
       iBit != endBits; ++iBit) {

    L1GlobalTriggerObjectMap const* objMap = gtObjectMapRecord->getObjectMap(*iBit);
    std::string algoName1 = objMap->algoName();

    if (algoName1 != algoNames2.at(*iBit)) {
      cms::Exception ex("L1GlobalTrigger");
      ex << "Algorithm names do not match";
      ex.addContext("Calling CompareToObjectMapRecord::analyze");
      throw ex;
    }

    // Now test the algorithm results
    if (objMap->algoGtlResult() != gtObjectMaps->algorithmResult(*iBit)) {
      cms::Exception ex("L1GlobalTrigger");
      ex << "Algorithm results do not match";
      ex.addContext("Calling CompareToObjectMapRecord::analyze");
      throw ex;
    }

    std::vector<L1GtLogicParser::OperandToken> const& operandTokens1 =
      objMap->operandTokenVector();

    L1GlobalTriggerObjectMaps::ConditionsInAlgorithm conditions2 = gtObjectMaps->getConditionsInAlgorithm(*iBit);
    if (conditions2.nConditions() != operandTokens1.size()) {
      cms::Exception ex("L1GlobalTrigger");
      ex << "Number of conditions does not match";
      ex.addContext("Calling CompareToObjectMapRecord::analyze");
      throw ex;
    }

    std::vector<std::string> conditionNames2;
    conditionNames2 = pset->getParameter<std::vector<std::string> >(algoNames2.at(*iBit));

    // Print out data from L1GlobalTriggerObjectMaps and ParameterSet registry
    if (verbose_) {
      std::cout << *iBit
                << "  " << algoNames2[*iBit]
                << "  " << gtObjectMaps->algorithmResult(*iBit) << "\n";

      for (unsigned j = 0; j < gtObjectMaps->getNumberOfConditions(*iBit); ++j) {
        L1GlobalTriggerObjectMaps::ConditionsInAlgorithm conditions = gtObjectMaps->getConditionsInAlgorithm(*iBit);
        std::cout << "    " << j
                  << "  " << conditionNames2[j]
                  << "  " << conditions.getConditionResult(j) << "\n";
        L1GlobalTriggerObjectMaps::CombinationsInCondition combinations = gtObjectMaps->getCombinationsInCondition(*iBit, j);
        for (unsigned m = 0; m < combinations.nCombinations(); ++m) {
          std::cout << "    "; 
          for (unsigned n = 0; n < combinations.nObjectsPerCombination(); ++n) {
            std::cout << "  " << static_cast<unsigned>(combinations.getObjectIndex(m,n)); 
          }
          std::cout << "\n";
        }
      }
    }

    // Loop over conditions
    unsigned iCondition = 0;
    for (std::vector<L1GtLogicParser::OperandToken>::const_iterator iToken1 = operandTokens1.begin(),
                                                                 endTokens1 = operandTokens1.end();
         iToken1 != endTokens1;
         ++iToken1, ++iCondition) {

      // Compare condition numbers
      if (iToken1->tokenNumber != static_cast<int>(iCondition)) {
        cms::Exception ex("L1GlobalTrigger");
        ex << "Token numbers not sequential";
        ex.addContext("Calling CompareToObjectMapRecord::analyze");
        throw ex;
      }

      // Compare condition names
      if (iToken1->tokenResult != conditions2.getConditionResult(iCondition)) {
        cms::Exception ex("L1GlobalTrigger");
        ex << "Condition results do not match";
        ex.addContext("Calling CompareToObjectMapRecord::analyze");
        throw ex;
      }

      // Compare condition names
      if (iToken1->tokenName != conditionNames2.at(iCondition)) {
        cms::Exception ex("L1GlobalTrigger");
        ex << "Condition names do not match";
        ex.addContext("Calling CompareToObjectMapRecord::analyze");
        throw ex;
      }

      // Compare the combinations of Indexes into the L1 collections
      L1GlobalTriggerObjectMaps::CombinationsInCondition combinations2 =
        gtObjectMaps->getCombinationsInCondition(*iBit, iCondition);

      CombinationsInCond const* combinations1 = objMap->getCombinationsInCond(iToken1->tokenNumber);
      if (combinations1->size() != combinations2.nCombinations()) {
        cms::Exception ex("L1GlobalTrigger");
        ex << "The number of combinations in a condition does not match";
        ex.addContext("Calling CompareToObjectMapRecord::analyze");
        throw ex;        
      }

      for (CombinationsInCond::const_iterator iCombo = combinations1->begin(),
                                           endCombos = combinations1->end();
           iCombo != endCombos; ++iCombo) {
        if (iCombo->size() != combinations2.nObjectsPerCombination()) {
          cms::Exception ex("L1GlobalTrigger");
          ex << "The number of indexes in a combination does not match";
          ex.addContext("Calling CompareToObjectMapRecord::analyze");
          throw ex;        
        }

        for (std::vector<int>::const_iterator iIndex = iCombo->begin(),
                                          endIndexes = iCombo->end();
             iIndex != endIndexes; ++iIndex) {

          if (*iIndex != combinations2.getObjectIndex(iCombo - combinations1->begin(), iIndex - iCombo->begin())) {
            cms::Exception ex("L1GlobalTrigger");
            ex << "Object index does not match";
            ex.addContext("Calling CompareToObjectMapRecord::analyze");
            throw ex;        
          }
        }
      }
    }
  }
}

/**
 * \class TriggerMenu
 *
 *
 * Description: L1 trigger menu.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *          Vladimir Rekovic - extend for overlap removal
 *          Elisa Fontanesi - extended for three-body correlation conditions
 *
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "L1Trigger/L1TGlobal/interface/TriggerMenu.h"

// system include files
#include <iostream>
#include <iomanip>

// user include files
#include "L1Trigger/L1TGlobal/interface/GlobalCondition.h"

// forward declarations

// constructor
TriggerMenu::TriggerMenu()
    : m_triggerMenuInterface("NULL"),
      m_triggerMenuName("NULL"),
      m_triggerMenuImplementation(0x0),
      m_scaleDbKey("NULL") {
  // empty
}

TriggerMenu::TriggerMenu(
    const std::string& triggerMenuNameVal,
    const unsigned int numberConditionChips,
    const std::vector<std::vector<MuonTemplate> >& vecMuonTemplateVal,
    const std::vector<std::vector<MuonShowerTemplate> >& vecMuonShowerTemplateVal,
    const std::vector<std::vector<CaloTemplate> >& vecCaloTemplateVal,
    const std::vector<std::vector<EnergySumTemplate> >& vecEnergySumTemplateVal,
    const std::vector<std::vector<EnergySumZdcTemplate> >& vecEnergySumZdcTemplateVal,
    const std::vector<std::vector<AXOL1TLTemplate> >& vecAXOL1TLTemplateVal,
    const std::vector<std::vector<ExternalTemplate> >& vecExternalTemplateVal,
    const std::vector<std::vector<CorrelationTemplate> >& vecCorrelationTemplateVal,
    const std::vector<std::vector<CorrelationThreeBodyTemplate> >& vecCorrelationThreeBodyTemplateVal,
    const std::vector<std::vector<CorrelationWithOverlapRemovalTemplate> >& vecCorrelationWithOverlapRemovalTemplateVal,
    const std::vector<std::vector<MuonTemplate> >& corMuonTemplateVal,
    const std::vector<std::vector<CaloTemplate> >& corCaloTemplateVal,
    const std::vector<std::vector<EnergySumTemplate> >& corEnergySumTemplateVal

    )
    : m_triggerMenuInterface("NULL"),
      m_triggerMenuName(triggerMenuNameVal),
      m_triggerMenuImplementation(0x0),
      m_scaleDbKey("NULL"),
      m_vecMuonTemplate(vecMuonTemplateVal),
      m_vecMuonShowerTemplate(vecMuonShowerTemplateVal),
      m_vecCaloTemplate(vecCaloTemplateVal),
      m_vecEnergySumTemplate(vecEnergySumTemplateVal),
      m_vecEnergySumZdcTemplate(vecEnergySumZdcTemplateVal),
      m_vecAXOL1TLTemplate(vecAXOL1TLTemplateVal),
      m_vecExternalTemplate(vecExternalTemplateVal),
      m_vecCorrelationTemplate(vecCorrelationTemplateVal),
      m_vecCorrelationThreeBodyTemplate(vecCorrelationThreeBodyTemplateVal),
      m_vecCorrelationWithOverlapRemovalTemplate(vecCorrelationWithOverlapRemovalTemplateVal),
      m_corMuonTemplate(corMuonTemplateVal),
      m_corCaloTemplate(corCaloTemplateVal),
      m_corEnergySumTemplate(corEnergySumTemplateVal) {
  m_conditionMap.resize(numberConditionChips);
  m_triggerMenuUUID = 0;
  buildGtConditionMap();
}

// copy constructor
TriggerMenu::TriggerMenu(const TriggerMenu& rhs) {
  m_triggerMenuInterface = rhs.m_triggerMenuInterface;
  m_triggerMenuName = rhs.m_triggerMenuName;
  m_triggerMenuImplementation = rhs.m_triggerMenuImplementation;
  m_scaleDbKey = rhs.m_scaleDbKey;
  m_triggerMenuUUID = rhs.m_triggerMenuUUID;

  // copy physics conditions
  m_vecMuonTemplate = rhs.m_vecMuonTemplate;
  m_vecMuonShowerTemplate = rhs.m_vecMuonShowerTemplate;
  m_vecCaloTemplate = rhs.m_vecCaloTemplate;
  m_vecEnergySumTemplate = rhs.m_vecEnergySumTemplate;
  m_vecEnergySumZdcTemplate = rhs.m_vecEnergySumZdcTemplate;
  m_vecAXOL1TLTemplate = rhs.m_vecAXOL1TLTemplate;
  m_vecExternalTemplate = rhs.m_vecExternalTemplate;

  m_vecCorrelationTemplate = rhs.m_vecCorrelationTemplate;
  m_vecCorrelationThreeBodyTemplate = rhs.m_vecCorrelationThreeBodyTemplate;
  m_vecCorrelationWithOverlapRemovalTemplate = rhs.m_vecCorrelationWithOverlapRemovalTemplate;
  m_corMuonTemplate = rhs.m_corMuonTemplate;
  m_corCaloTemplate = rhs.m_corCaloTemplate;
  m_corEnergySumTemplate = rhs.m_corEnergySumTemplate;

  // rebuild condition map to update the pointers
  // (only physics conditions are included in it)
  m_conditionMap.resize(rhs.m_conditionMap.size());
  (*this).buildGtConditionMap();

  // copy algorithm map
  m_algorithmMap = rhs.m_algorithmMap;
  m_algorithmAliasMap = rhs.m_algorithmAliasMap;

  // copy technical triggers
  // (separate map for technical triggers and physics triggers)
  //m_technicalTriggerMap = rhs.m_technicalTriggerMap;
}

// destructor
TriggerMenu::~TriggerMenu() {
  // loop over condition maps (one map per condition chip)
  for (std::vector<l1t::ConditionMap>::iterator itCondOnChip = m_conditionMap.begin();
       itCondOnChip != m_conditionMap.end();
       itCondOnChip++) {
    itCondOnChip->clear();
  }

  m_algorithmMap.clear();
  m_algorithmAliasMap.clear();
}

// assignment operator
TriggerMenu& TriggerMenu::operator=(const TriggerMenu& rhs) {
  if (this != &rhs) {
    m_triggerMenuInterface = rhs.m_triggerMenuInterface;
    m_triggerMenuName = rhs.m_triggerMenuName;
    m_triggerMenuImplementation = rhs.m_triggerMenuImplementation;
    m_triggerMenuUUID = rhs.m_triggerMenuUUID;

    m_vecMuonTemplate = rhs.m_vecMuonTemplate;
    m_vecMuonShowerTemplate = rhs.m_vecMuonShowerTemplate;
    m_vecCaloTemplate = rhs.m_vecCaloTemplate;
    m_vecEnergySumTemplate = rhs.m_vecEnergySumTemplate;
    m_vecEnergySumZdcTemplate = rhs.m_vecEnergySumZdcTemplate;
    m_vecAXOL1TLTemplate = rhs.m_vecAXOL1TLTemplate;
    m_vecExternalTemplate = rhs.m_vecExternalTemplate;

    m_vecCorrelationTemplate = rhs.m_vecCorrelationTemplate;
    m_vecCorrelationThreeBodyTemplate = rhs.m_vecCorrelationThreeBodyTemplate;
    m_vecCorrelationWithOverlapRemovalTemplate = rhs.m_vecCorrelationWithOverlapRemovalTemplate;
    m_corMuonTemplate = rhs.m_corMuonTemplate;
    m_corCaloTemplate = rhs.m_corCaloTemplate;
    m_corEnergySumTemplate = rhs.m_corEnergySumTemplate;

    m_algorithmMap = rhs.m_algorithmMap;
    m_algorithmAliasMap = rhs.m_algorithmAliasMap;

    //        m_technicalTriggerMap = rhs.m_technicalTriggerMap;
  }

  // rebuild condition map to update the pointers
  // (only physics conditions are included in it)
  m_conditionMap.resize(rhs.m_conditionMap.size());
  (*this).buildGtConditionMap();

  // return the object
  return *this;
}

// set the condition maps
void TriggerMenu::setGtConditionMap(const std::vector<l1t::ConditionMap>& condMap) { m_conditionMap = condMap; }

// build the condition maps
void TriggerMenu::buildGtConditionMap() {
  // clear the conditions from the maps, if any
  for (std::vector<l1t::ConditionMap>::iterator itCondOnChip = m_conditionMap.begin();
       itCondOnChip != m_conditionMap.end();
       itCondOnChip++) {
    itCondOnChip->clear();
  }

  // always check that the size of the condition map is greater than the size
  // of the specific condition vector
  size_t condMapSize = m_conditionMap.size();

  //
  size_t vecMuonSize = m_vecMuonTemplate.size();
  if (condMapSize < vecMuonSize) {
    m_conditionMap.resize(vecMuonSize);
    condMapSize = m_conditionMap.size();
  }

  int chipNr = -1;

  for (std::vector<std::vector<MuonTemplate> >::iterator itCondOnChip = m_vecMuonTemplate.begin();
       itCondOnChip != m_vecMuonTemplate.end();
       itCondOnChip++) {
    chipNr++;

    for (std::vector<MuonTemplate>::iterator itCond = itCondOnChip->begin(); itCond != itCondOnChip->end(); itCond++) {
      (m_conditionMap.at(chipNr))[itCond->condName()] = &(*itCond);
    }
  }

  //
  size_t vecMuonShowerSize = m_vecMuonShowerTemplate.size();
  if (condMapSize < vecMuonShowerSize) {
    m_conditionMap.resize(vecMuonShowerSize);
    condMapSize = m_conditionMap.size();
  }

  chipNr = -1;

  for (std::vector<std::vector<MuonShowerTemplate> >::iterator itCondOnChip = m_vecMuonShowerTemplate.begin();
       itCondOnChip != m_vecMuonShowerTemplate.end();
       itCondOnChip++) {
    chipNr++;

    for (std::vector<MuonShowerTemplate>::iterator itCond = itCondOnChip->begin(); itCond != itCondOnChip->end();
         itCond++) {
      (m_conditionMap.at(chipNr))[itCond->condName()] = &(*itCond);
    }
  }

  //
  size_t vecCaloSize = m_vecCaloTemplate.size();
  if (condMapSize < vecCaloSize) {
    m_conditionMap.resize(vecCaloSize);
    condMapSize = m_conditionMap.size();
  }

  chipNr = -1;
  for (std::vector<std::vector<CaloTemplate> >::iterator itCondOnChip = m_vecCaloTemplate.begin();
       itCondOnChip != m_vecCaloTemplate.end();
       itCondOnChip++) {
    
    chipNr++;

    for (std::vector<CaloTemplate>::iterator itCond = itCondOnChip->begin(); itCond != itCondOnChip->end(); itCond++) {
      (m_conditionMap.at(chipNr))[itCond->condName()] = &(*itCond);
    }
  }

  //
  size_t vecEnergySumSize = m_vecEnergySumTemplate.size();
  if (condMapSize < vecEnergySumSize) {
    m_conditionMap.resize(vecEnergySumSize);
    condMapSize = m_conditionMap.size();
  }

  chipNr = -1;
  for (std::vector<std::vector<EnergySumTemplate> >::iterator itCondOnChip = m_vecEnergySumTemplate.begin();
       itCondOnChip != m_vecEnergySumTemplate.end();
       itCondOnChip++) {
    chipNr++;

    for (std::vector<EnergySumTemplate>::iterator itCond = itCondOnChip->begin(); itCond != itCondOnChip->end();
         itCond++) {
      (m_conditionMap.at(chipNr))[itCond->condName()] = &(*itCond);
    }
  }

  //
  size_t vecEnergySumZdcSize = m_vecEnergySumZdcTemplate.size();
  if (condMapSize < vecEnergySumZdcSize) {
    m_conditionMap.resize(vecEnergySumZdcSize);
    condMapSize = m_conditionMap.size();
  }

  chipNr = -1;
  for (std::vector<std::vector<EnergySumZdcTemplate> >::iterator itCondOnChip = m_vecEnergySumZdcTemplate.begin();
       itCondOnChip != m_vecEnergySumZdcTemplate.end();
       itCondOnChip++) {
    chipNr++;

    for (std::vector<EnergySumZdcTemplate>::iterator itCond = itCondOnChip->begin(); itCond != itCondOnChip->end();
         itCond++) {
      (m_conditionMap.at(chipNr))[itCond->condName()] = &(*itCond);
    }
  }

  //
  size_t vecAXOL1TLSize = m_vecAXOL1TLTemplate.size();
  if (condMapSize < vecAXOL1TLSize) {
    m_conditionMap.resize(vecAXOL1TLSize);
    condMapSize = m_conditionMap.size();
  }

  chipNr = -1;

  for (std::vector<std::vector<AXOL1TLTemplate> >::iterator itCondOnChip = m_vecAXOL1TLTemplate.begin();
       itCondOnChip != m_vecAXOL1TLTemplate.end();
       itCondOnChip++) {
    chipNr++;

    for (std::vector<AXOL1TLTemplate>::iterator itCond = itCondOnChip->begin(); itCond != itCondOnChip->end();
         itCond++) {
      (m_conditionMap.at(chipNr))[itCond->condName()] = &(*itCond);
    }
  }

  
  /// DMP: Comment out unused templates for now
  //
  //
  size_t vecExternalSize = m_vecExternalTemplate.size();
  if (condMapSize < vecExternalSize) {
    m_conditionMap.resize(vecExternalSize);
    condMapSize = m_conditionMap.size();
  }

  chipNr = -1;
  for (std::vector<std::vector<ExternalTemplate> >::iterator itCondOnChip = m_vecExternalTemplate.begin();
       itCondOnChip != m_vecExternalTemplate.end();
       itCondOnChip++) {
    chipNr++;

    for (std::vector<ExternalTemplate>::iterator itCond = itCondOnChip->begin(); itCond != itCondOnChip->end();
         itCond++) {
      (m_conditionMap.at(chipNr))[itCond->condName()] = &(*itCond);
    }
  }

  //
  size_t vecCorrelationSize = m_vecCorrelationTemplate.size();
  if (condMapSize < vecCorrelationSize) {
    m_conditionMap.resize(vecCorrelationSize);
    condMapSize = m_conditionMap.size();
  }

  chipNr = -1;
  for (std::vector<std::vector<CorrelationTemplate> >::iterator itCondOnChip = m_vecCorrelationTemplate.begin();
       itCondOnChip != m_vecCorrelationTemplate.end();
       itCondOnChip++) {
    chipNr++;

    for (std::vector<CorrelationTemplate>::iterator itCond = itCondOnChip->begin(); itCond != itCondOnChip->end();
         itCond++) {
      (m_conditionMap.at(chipNr))[itCond->condName()] = &(*itCond);
    }
  }

  //
  size_t vecCorrelationThreeBodySize = m_vecCorrelationThreeBodyTemplate.size();
  if (condMapSize < vecCorrelationThreeBodySize) {
    m_conditionMap.resize(vecCorrelationThreeBodySize);
    condMapSize = m_conditionMap.size();
  }

  chipNr = -1;
  for (std::vector<std::vector<CorrelationThreeBodyTemplate> >::iterator itCondOnChip =
           m_vecCorrelationThreeBodyTemplate.begin();
       itCondOnChip != m_vecCorrelationThreeBodyTemplate.end();
       itCondOnChip++) {
    chipNr++;

    for (std::vector<CorrelationThreeBodyTemplate>::iterator itCond = itCondOnChip->begin();
         itCond != itCondOnChip->end();
         itCond++) {
      (m_conditionMap.at(chipNr))[itCond->condName()] = &(*itCond);
    }
  }

  //
  size_t vecCorrelationWORSize = m_vecCorrelationWithOverlapRemovalTemplate.size();
  if (condMapSize < vecCorrelationWORSize) {
    m_conditionMap.resize(vecCorrelationWORSize);
    condMapSize = m_conditionMap.size();
  }

  chipNr = -1;
  for (std::vector<std::vector<CorrelationWithOverlapRemovalTemplate> >::iterator itCondOnChip =
           m_vecCorrelationWithOverlapRemovalTemplate.begin();
       itCondOnChip != m_vecCorrelationWithOverlapRemovalTemplate.end();
       itCondOnChip++) {
    chipNr++;

    for (std::vector<CorrelationWithOverlapRemovalTemplate>::iterator itCond = itCondOnChip->begin();
         itCond != itCondOnChip->end();
         itCond++) {
      (m_conditionMap.at(chipNr))[itCond->condName()] = &(*itCond);
    }
  }
}

// set the trigger menu name
void TriggerMenu::setGtTriggerMenuInterface(const std::string& menuInterface) {
  m_triggerMenuInterface = menuInterface;
}

void TriggerMenu::setGtTriggerMenuName(const std::string& menuName) { m_triggerMenuName = menuName; }

void TriggerMenu::setGtTriggerMenuImplementation(const unsigned long menuImplementation) {
  m_triggerMenuImplementation = menuImplementation;
}

void TriggerMenu::setGtTriggerMenuUUID(const unsigned long uuid) { m_triggerMenuUUID = uuid; }

// set menu associated scale key
void TriggerMenu::setGtScaleDbKey(const std::string& scaleKey) { m_scaleDbKey = scaleKey; }

// set menu associated scale key
void TriggerMenu::setGtScales(const l1t::GlobalScales& scales) { m_gtScales = scales; }

// get / set the vectors containing the conditions
void TriggerMenu::setVecMuonTemplate(const std::vector<std::vector<MuonTemplate> >& vecMuonTempl) {
  m_vecMuonTemplate = vecMuonTempl;
}

void TriggerMenu::setVecCaloTemplate(const std::vector<std::vector<CaloTemplate> >& vecCaloTempl) {
  m_vecCaloTemplate = vecCaloTempl;
}

void TriggerMenu::setVecEnergySumTemplate(const std::vector<std::vector<EnergySumTemplate> >& vecEnergySumTempl) {
  m_vecEnergySumTemplate = vecEnergySumTempl;
}

void TriggerMenu::setVecEnergySumZdcTemplate(
    const std::vector<std::vector<EnergySumZdcTemplate> >& vecEnergySumZdcTempl) {
  m_vecEnergySumZdcTemplate = vecEnergySumZdcTempl;
}

void TriggerMenu::setVecAXOL1TLTemplate(const std::vector<std::vector<AXOL1TLTemplate> >& vecAXOL1TLTempl) { 
  m_vecAXOL1TLTemplate = vecAXOL1TLTempl;
}

void TriggerMenu::setVecExternalTemplate(const std::vector<std::vector<ExternalTemplate> >& vecExternalTempl) {
  m_vecExternalTemplate = vecExternalTempl;
}

void TriggerMenu::setVecCorrelationTemplate(const std::vector<std::vector<CorrelationTemplate> >& vecCorrelationTempl) {
  m_vecCorrelationTemplate = vecCorrelationTempl;
}

void TriggerMenu::setVecCorrelationThreeBodyTemplate(
    const std::vector<std::vector<CorrelationThreeBodyTemplate> >& vecCorrelationThreeBodyTempl) {
  m_vecCorrelationThreeBodyTemplate = vecCorrelationThreeBodyTempl;
}

void TriggerMenu::setVecCorrelationWithOverlapRemovalTemplate(
    const std::vector<std::vector<CorrelationWithOverlapRemovalTemplate> >& vecCorrelationTempl) {
  m_vecCorrelationWithOverlapRemovalTemplate = vecCorrelationTempl;
}

// set the vectors containing the conditions for correlation templates
void TriggerMenu::setCorMuonTemplate(const std::vector<std::vector<MuonTemplate> >& corMuonTempl) {
  m_corMuonTemplate = corMuonTempl;
}

void TriggerMenu::setCorCaloTemplate(const std::vector<std::vector<CaloTemplate> >& corCaloTempl) {
  m_corCaloTemplate = corCaloTempl;
}

void TriggerMenu::setCorEnergySumTemplate(const std::vector<std::vector<EnergySumTemplate> >& corEnergySumTempl) {
  m_corEnergySumTemplate = corEnergySumTempl;
}

// set the algorithm map (by algorithm names)
void TriggerMenu::setGtAlgorithmMap(const l1t::AlgorithmMap& algoMap) { m_algorithmMap = algoMap; }

// set the algorithm map (by algorithm aliases)
void TriggerMenu::setGtAlgorithmAliasMap(const l1t::AlgorithmMap& algoMap) { m_algorithmAliasMap = algoMap; }

/*
// set the technical trigger map
void TriggerMenu::setGtTechnicalTriggerMap(const l1t::AlgorithmMap& ttMap) {
    m_technicalTriggerMap = ttMap;
}
*/

// print the trigger menu (bit number, algorithm name, logical expression)
void TriggerMenu::print(std::ostream& myCout, int& printVerbosity) const {
  // use another map <int, GlobalAlgorithm> to get the menu sorted after bit number
  // both algorithm and bit numbers are unique
  std::map<int, const GlobalAlgorithm*> algoBitToAlgo;
  typedef std::map<int, const GlobalAlgorithm*>::const_iterator CItBit;

  for (l1t::CItAlgo itAlgo = m_algorithmMap.begin(); itAlgo != m_algorithmMap.end(); itAlgo++) {
    int bitNumber = (itAlgo->second).algoBitNumber();
    algoBitToAlgo[bitNumber] = &(itAlgo->second);
  }

  size_t nrDefinedAlgo = algoBitToAlgo.size();

  /*
    // idem for technical trigger map - only name and bit number are relevant for them
    std::map<int, const GlobalAlgorithm*> ttBitToTt;

    for (l1t::CItAlgo itAlgo = m_technicalTriggerMap.begin(); itAlgo
            != m_technicalTriggerMap.end(); itAlgo++) {

        int bitNumber = (itAlgo->second).algoBitNumber();
        ttBitToTt[bitNumber] = &(itAlgo->second);
    }

    size_t nrDefinedTechTrig = ttBitToTt.size();
*/
  //

  switch (printVerbosity) {
    case 0: {
      // header for printing algorithms

      myCout << "\n   ********** L1 Trigger Menu - printing   ********** \n"
             << "\nL1 Trigger Menu Interface:         " << m_triggerMenuInterface
             << "\nL1 Trigger Menu Name:              " << m_triggerMenuName << "\nL1 Trigger Menu UUID (hash):     0x"
             << std::hex << m_triggerMenuUUID << std::dec << "\nL1 Trigger Menu Firmware (hash): 0x" << std::hex
             << m_triggerMenuImplementation << std::dec << "\nAssociated Scale DB Key: " << m_scaleDbKey << "\n\n"
             << "\nL1 Physics Algorithms: " << nrDefinedAlgo << " algorithms defined."
             << "\n\n"
             << "Bit Number " << std::right << std::setw(35) << "Algorithm Name"
             << "  " << std::right << std::setw(35) << "Algorithm Alias" << std::endl;

      for (CItBit itBit = algoBitToAlgo.begin(); itBit != algoBitToAlgo.end(); itBit++) {
        int bitNumber = itBit->first;
        std::string aName = (itBit->second)->algoName();
        std::string aAlias = (itBit->second)->algoAlias();

        myCout << std::setw(6) << bitNumber << "     " << std::right << std::setw(35) << aName << "  " << std::right
               << std::setw(35) << aAlias << std::endl;
      }
      /*
            myCout
            << "\nL1 Technical Triggers: " << nrDefinedTechTrig
            << " technical triggers defined." << "\n\n" << std::endl;
            if (nrDefinedTechTrig) {
                myCout << "Bit Number " << " Technical trigger name " << std::endl;
            }

            for (CItBit itBit = ttBitToTt.begin(); itBit != ttBitToTt.end(); itBit++) {

                int bitNumber = itBit->first;
                std::string aName = (itBit->second)->algoName();
                std::string aAlias = (itBit->second)->algoAlias();

                myCout << std::setw(6) << bitNumber << "       "
                << std::right << std::setw(35) << aName << "  "
                << std::right << std::setw(35) << aAlias
                << std::endl;
            }
*/
    } break;

    case 1: {
      // header for printing algorithms

      myCout << "\n   ********** L1 Trigger Menu - printing   ********** \n"
             << "\nL1 Trigger Menu Interface:         " << m_triggerMenuInterface
             << "\nL1 Trigger Menu Name:              " << m_triggerMenuName << "\nL1 Trigger Menu UUID (hash):     0x"
             << std::hex << m_triggerMenuUUID << std::dec << "\nL1 Trigger Menu Firmware (hash): 0x" << std::hex
             << m_triggerMenuImplementation << std::dec << "\nAssociated Scale DB Key: " << m_scaleDbKey << "\n\n"
             << "\nL1 Physics Algorithms: " << nrDefinedAlgo << " algorithms defined."
             << "\n\n"
             << "Bit Number " << std::right << std::setw(35) << "Algorithm Name"
             << "  " << std::right << std::setw(35) << "Algorithm Alias"
             << "\n  Logical Expression \n"
             << std::endl;

      for (CItBit itBit = algoBitToAlgo.begin(); itBit != algoBitToAlgo.end(); itBit++) {
        int bitNumber = itBit->first;
        std::string aName = (itBit->second)->algoName();
        std::string aAlias = (itBit->second)->algoAlias();
        std::string aLogicalExpression = (itBit->second)->algoLogicalExpression();

        myCout << std::setw(6) << bitNumber << "     " << std::right << std::setw(35) << aName << "  " << std::right
               << std::setw(35) << aAlias << "\n  Logical expression: " << aLogicalExpression << "\n"
               << std::endl;
      }
      /*
            myCout
            << "\nL1 Technical Triggers: " << nrDefinedTechTrig
            << " technical triggers defined." << "\n\n" << std::endl;
            if (nrDefinedTechTrig) {
                myCout << "Bit Number " << " Technical trigger name " << std::endl;
            }

            for (CItBit itBit = ttBitToTt.begin(); itBit != ttBitToTt.end(); itBit++) {

                int bitNumber = itBit->first;
                std::string aName = (itBit->second)->algoName();

                myCout << std::setw(6) << bitNumber << "       " << aName << std::endl;
            }
*/
    } break;

    case 2: {
      // header for printing algorithms

      myCout << "\n   ********** L1 Trigger Menu - printing   ********** \n"
             << "\nL1 Trigger Menu Interface:         " << m_triggerMenuInterface
             << "\nL1 Trigger Menu Name:              " << m_triggerMenuName << "\nL1 Trigger Menu UUID (hash):     0x"
             << std::hex << m_triggerMenuUUID << std::dec << "\nL1 Trigger Menu Firmware (hash): 0x" << std::hex
             << m_triggerMenuImplementation << std::dec << "\nAssociated Scale DB Key: " << m_scaleDbKey << "\n\n"
             << "\nL1 Physics Algorithms: " << nrDefinedAlgo << " algorithms defined."
             << "\n\n"
             << std::endl;

      for (CItBit itBit = algoBitToAlgo.begin(); itBit != algoBitToAlgo.end(); itBit++) {
        (itBit->second)->print(myCout);
      }

      myCout << "\nNumber of condition chips: " << m_conditionMap.size() << "\n" << std::endl;

      int chipNr = -1;
      int totalNrConditions = 0;

      for (std::vector<l1t::ConditionMap>::const_iterator itCondOnChip = m_conditionMap.begin();
           itCondOnChip != m_conditionMap.end();
           itCondOnChip++) {
        chipNr++;

        int condMapSize = itCondOnChip->size();
        totalNrConditions += condMapSize;

        myCout << "\nTotal number of conditions on condition chip " << chipNr << ": " << condMapSize << " conditions.\n"
               << std::endl;

        for (l1t::CItCond itCond = itCondOnChip->begin(); itCond != itCondOnChip->end(); itCond++) {
          (itCond->second)->print(myCout);
        }
      }

      myCout << "\nTotal number of conditions on all condition chips: " << totalNrConditions << "\n" << std::endl;
      /*
            myCout
            << "\nL1 Technical Triggers: " << nrDefinedTechTrig
            << " technical triggers defined." << "\n\n" << std::endl;
            if (nrDefinedTechTrig) {
                myCout << "Bit Number " << " Technical trigger name " << std::endl;
            }

            for (CItBit itBit = ttBitToTt.begin(); itBit != ttBitToTt.end(); itBit++) {

                int bitNumber = itBit->first;
                std::string aName = (itBit->second)->algoName();

                myCout << std::setw(6) << bitNumber << "       " << aName << std::endl;
            }
*/

    } break;

    default: {
      myCout << "\n   ********** L1 Trigger Menu - printing   ********** \n\n"
             << "Verbosity level: " << printVerbosity << " not implemented.\n\n"
             << std::endl;
    } break;
  }
}

// get the result for algorithm with name algName
// use directly the format of decisionWord (no typedef)
const bool TriggerMenu::gtAlgorithmResult(const std::string& algName, const std::vector<bool>& decWord) const {
  bool algResult = false;

  l1t::CItAlgo itAlgo = m_algorithmMap.find(algName);
  if (itAlgo != m_algorithmMap.end()) {
    int bitNumber = (itAlgo->second).algoBitNumber();
    algResult = decWord.at(bitNumber);
    return algResult;
  }

  // return false if the algorithm name is not found in the menu
  // TODO throw exception or LogInfo would be better - but the class is used in
  // XDAQ Trigger Supervisor (outside CMSSW) hence no CMSSW dependence
  // is allowed here...

  return false;
}

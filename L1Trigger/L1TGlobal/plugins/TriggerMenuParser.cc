/**
 * \class TriggerMenuParser
 *
 *
 * Description: Xerces-C XML parser for the L1 Trigger menu.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \orig author: Vasile Mihai Ghete - HEPHY Vienna
 *
 * \new features: Vladimir Rekovic
 *                - indexing
 *                - correlations with overlap object removal
 * \new features: Richard Cavanaugh
 *                - LLP displaced muons
 *                - LLP displaced jets
 * \new features: Elisa Fontanesi
 *                - extended for three-body correlation conditions
 * \new features: Dragana Pilipovic
 *                - updated for invariant mass over delta R condition
 * \new features: Bernhard Arnold, Elisa Fontanesi
 *                - extended for muon track finder index feature (used for Run 3 muon monitoring seeds)
 *                - checkRangeEta function allows to use up to five eta cuts in L1 algorithms
 * \new features: Elisa Fontanesi
 *                - extended for Zero Degree Calorimeter triggers (used for Run 3 HI data-taking)
 * \new features: Melissa Quinnan, Elisa Fontanesi
 *                - extended for AXOL1TL anomaly detection triggers (used for Run 3 data-taking)
 *
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "TriggerMenuParser.h"

// system include files
#include <string>
#include <vector>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

#include "L1Trigger/L1TGlobal/interface/GlobalCondition.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include "tmEventSetup/tmEventSetup.hh"
#include "tmEventSetup/esTypes.hh"

#include "CondFormats/L1TObjects/interface/L1TUtmTriggerMenu.h"
#include "CondFormats/L1TObjects/interface/L1TUtmAlgorithm.h"
#include "CondFormats/L1TObjects/interface/L1TUtmCondition.h"
#include "CondFormats/L1TObjects/interface/L1TUtmObject.h"
#include "CondFormats/L1TObjects/interface/L1TUtmCut.h"
#include "CondFormats/L1TObjects/interface/L1TUtmScale.h"
#include "tmGrammar/Algorithm.hh"

#include <cstdint>

// constructor
l1t::TriggerMenuParser::TriggerMenuParser()
    : m_triggerMenuInterface("NULL"),
      m_triggerMenuName("NULL"),
      m_triggerMenuImplementation(0x0),
      m_scaleDbKey("NULL")

{
  // menu names, scale key initialized to NULL due to ORACLE treatment of strings

  // empty
}

// destructor
l1t::TriggerMenuParser::~TriggerMenuParser() { clearMaps(); }

// set the number of condition chips in GTL
void l1t::TriggerMenuParser::setGtNumberConditionChips(const unsigned int& numberConditionChipsValue) {
  m_numberConditionChips = numberConditionChipsValue;
}

// set the number of pins on the GTL condition chips
void l1t::TriggerMenuParser::setGtPinsOnConditionChip(const unsigned int& pinsOnConditionChipValue) {
  m_pinsOnConditionChip = pinsOnConditionChipValue;
}

// set the correspondence "condition chip - GTL algorithm word"
// in the hardware
void l1t::TriggerMenuParser::setGtOrderConditionChip(const std::vector<int>& orderConditionChipValue) {
  m_orderConditionChip = orderConditionChipValue;
}

// set the number of physics trigger algorithms
void l1t::TriggerMenuParser::setGtNumberPhysTriggers(const unsigned int& numberPhysTriggersValue) {
  m_numberPhysTriggers = numberPhysTriggersValue;
}

// set the condition maps
void l1t::TriggerMenuParser::setGtConditionMap(const std::vector<ConditionMap>& condMap) { m_conditionMap = condMap; }

// set the trigger menu name
void l1t::TriggerMenuParser::setGtTriggerMenuInterface(const std::string& menuInterface) {
  m_triggerMenuInterface = menuInterface;
}

// set the trigger menu uuid
void l1t::TriggerMenuParser::setGtTriggerMenuUUID(const int uuid) { m_triggerMenuUUID = uuid; }

void l1t::TriggerMenuParser::setGtTriggerMenuName(const std::string& menuName) { m_triggerMenuName = menuName; }

void l1t::TriggerMenuParser::setGtTriggerMenuImplementation(const unsigned long& menuImplementation) {
  m_triggerMenuImplementation = menuImplementation;
}

// set menu associated scale key
void l1t::TriggerMenuParser::setGtScaleDbKey(const std::string& scaleKey) { m_scaleDbKey = scaleKey; }

// set the vectors containing the conditions
void l1t::TriggerMenuParser::setVecMuonTemplate(const std::vector<std::vector<MuonTemplate> >& vecMuonTempl) {
  m_vecMuonTemplate = vecMuonTempl;
}

void l1t::TriggerMenuParser::setVecMuonShowerTemplate(
    const std::vector<std::vector<MuonShowerTemplate> >& vecMuonShowerTempl) {
  m_vecMuonShowerTemplate = vecMuonShowerTempl;
}

void l1t::TriggerMenuParser::setVecCaloTemplate(const std::vector<std::vector<CaloTemplate> >& vecCaloTempl) {
  m_vecCaloTemplate = vecCaloTempl;
}

void l1t::TriggerMenuParser::setVecEnergySumTemplate(
    const std::vector<std::vector<EnergySumTemplate> >& vecEnergySumTempl) {
  m_vecEnergySumTemplate = vecEnergySumTempl;
}

void l1t::TriggerMenuParser::setVecEnergySumZdcTemplate(
    const std::vector<std::vector<EnergySumZdcTemplate> >& vecEnergySumZdcTempl) {
  m_vecEnergySumZdcTemplate = vecEnergySumZdcTempl;
}

void l1t::TriggerMenuParser::setVecAXOL1TLTemplate(const std::vector<std::vector<AXOL1TLTemplate> >& vecAXOL1TLTempl) {
  m_vecAXOL1TLTemplate = vecAXOL1TLTempl;
}

void l1t::TriggerMenuParser::setVecExternalTemplate(
    const std::vector<std::vector<ExternalTemplate> >& vecExternalTempl) {
  m_vecExternalTemplate = vecExternalTempl;
}

void l1t::TriggerMenuParser::setVecCorrelationTemplate(
    const std::vector<std::vector<CorrelationTemplate> >& vecCorrelationTempl) {
  m_vecCorrelationTemplate = vecCorrelationTempl;
}

void l1t::TriggerMenuParser::setVecCorrelationThreeBodyTemplate(
    const std::vector<std::vector<CorrelationThreeBodyTemplate> >& vecCorrelationThreeBodyTempl) {
  m_vecCorrelationThreeBodyTemplate = vecCorrelationThreeBodyTempl;
}

void l1t::TriggerMenuParser::setVecCorrelationWithOverlapRemovalTemplate(
    const std::vector<std::vector<CorrelationWithOverlapRemovalTemplate> >& vecCorrelationWithOverlapRemovalTempl) {
  m_vecCorrelationWithOverlapRemovalTemplate = vecCorrelationWithOverlapRemovalTempl;
}

// set the vectors containing the conditions for correlation templates
//
void l1t::TriggerMenuParser::setCorMuonTemplate(const std::vector<std::vector<MuonTemplate> >& corMuonTempl) {
  m_corMuonTemplate = corMuonTempl;
}

void l1t::TriggerMenuParser::setCorCaloTemplate(const std::vector<std::vector<CaloTemplate> >& corCaloTempl) {
  m_corCaloTemplate = corCaloTempl;
}

void l1t::TriggerMenuParser::setCorEnergySumTemplate(
    const std::vector<std::vector<EnergySumTemplate> >& corEnergySumTempl) {
  m_corEnergySumTemplate = corEnergySumTempl;
}

// set the algorithm map (by algorithm names)
void l1t::TriggerMenuParser::setGtAlgorithmMap(const AlgorithmMap& algoMap) { m_algorithmMap = algoMap; }

// set the algorithm map (by algorithm aliases)
void l1t::TriggerMenuParser::setGtAlgorithmAliasMap(const AlgorithmMap& algoMap) { m_algorithmAliasMap = algoMap; }

std::map<std::string, unsigned int> l1t::TriggerMenuParser::getExternalSignals(const L1TUtmTriggerMenu* utmMenu) {
  using namespace tmeventsetup;
  const std::map<std::string, L1TUtmCondition>& condMap = utmMenu->getConditionMap();

  std::map<std::string, unsigned int> extBitMap;

  //loop over the algorithms
  for (const auto& cit : condMap) {
    const L1TUtmCondition& condition = cit.second;
    if (condition.getType() == esConditionType::Externals) {
      // Get object for External conditions
      const std::vector<L1TUtmObject>& objects = condition.getObjects();
      for (const auto& object : objects) {
        if (object.getType() == esObjectType::EXT) {
          unsigned int channelID = object.getExternalChannelId();
          std::string name = object.getExternalSignalName();

          if (extBitMap.count(name) == 0)
            extBitMap.insert(std::map<std::string, unsigned int>::value_type(name, channelID));
        }
      }
    }
  }

  return extBitMap;
}

// parse def.xml file
void l1t::TriggerMenuParser::parseCondFormats(const L1TUtmTriggerMenu* utmMenu) {
  // resize the vector of condition maps
  // the number of condition chips should be correctly set before calling parseXmlFile
  m_conditionMap.resize(m_numberConditionChips);

  m_vecMuonTemplate.resize(m_numberConditionChips);
  m_vecMuonShowerTemplate.resize(m_numberConditionChips);
  m_vecCaloTemplate.resize(m_numberConditionChips);
  m_vecEnergySumTemplate.resize(m_numberConditionChips);
  m_vecEnergySumZdcTemplate.resize(m_numberConditionChips);
  m_vecAXOL1TLTemplate.resize(m_numberConditionChips);
  m_vecExternalTemplate.resize(m_numberConditionChips);

  m_vecCorrelationTemplate.resize(m_numberConditionChips);
  m_vecCorrelationThreeBodyTemplate.resize(m_numberConditionChips);
  m_vecCorrelationWithOverlapRemovalTemplate.resize(m_numberConditionChips);
  m_corMuonTemplate.resize(m_numberConditionChips);
  m_corCaloTemplate.resize(m_numberConditionChips);
  m_corEnergySumTemplate.resize(m_numberConditionChips);

  using namespace tmeventsetup;
  using namespace Algorithm;

  //get the meta data
  m_triggerMenuDescription = utmMenu->getComment();
  m_triggerMenuDate = utmMenu->getDatetime();
  m_triggerMenuImplementation = (getMmHashN(utmMenu->getFirmwareUuid()) & 0xFFFFFFFF);  //make sure we only have 32 bits
  m_triggerMenuName = utmMenu->getName();
  m_triggerMenuInterface = utmMenu->getVersion();                     //BLW: correct descriptor?
  m_triggerMenuUUID = (getMmHashN(utmMenu->getName()) & 0xFFFFFFFF);  //make sure we only have 32 bits

  const std::map<std::string, L1TUtmAlgorithm>& algoMap = utmMenu->getAlgorithmMap();
  const std::map<std::string, L1TUtmCondition>& condMap = utmMenu->getConditionMap();
  //We use es types for scale map to use auxiliary functions without having to duplicate code
  const std::map<std::string, tmeventsetup::esScale> scaleMap(std::begin(utmMenu->getScaleMap()),
                                                              std::end(utmMenu->getScaleMap()));

  // parse the scales
  m_gtScales.setScalesName(utmMenu->getScaleSetName());
  parseScales(scaleMap);

  //loop over the algorithms
  for (const auto& cit : algoMap) {
    //condition chip (artifact)  TO DO: Update
    int chipNr = 0;

    //get algorithm
    const L1TUtmAlgorithm& algo = cit.second;

    //parse the algorithm
    parseAlgorithm(algo, chipNr);  //blw

    //get conditions for this algorithm
    const std::vector<std::string>& rpn_vec = algo.getRpnVector();
    for (size_t ii = 0; ii < rpn_vec.size(); ii++) {
      const std::string& token = rpn_vec.at(ii);
      if (isGate(token))
        continue;
      //      long hash = getHash(token);
      const L1TUtmCondition& condition = condMap.find(token)->second;

      //check to see if this condtion already exists
      if ((m_conditionMap[chipNr]).count(condition.getName()) == 0) {
        // parse Calo Conditions (EG, Jets, Taus)
        if (condition.getType() == esConditionType::SingleEgamma ||
            condition.getType() == esConditionType::DoubleEgamma ||
            condition.getType() == esConditionType::TripleEgamma ||
            condition.getType() == esConditionType::QuadEgamma || condition.getType() == esConditionType::SingleTau ||
            condition.getType() == esConditionType::DoubleTau || condition.getType() == esConditionType::TripleTau ||
            condition.getType() == esConditionType::QuadTau || condition.getType() == esConditionType::SingleJet ||
            condition.getType() == esConditionType::DoubleJet || condition.getType() == esConditionType::TripleJet ||
            condition.getType() == esConditionType::QuadJet) {
          parseCalo(condition, chipNr, false);

          // parse Energy Sums (and HI trigger objects, treated as energy sums or counters)
        } else if (condition.getType() == esConditionType::TotalEt ||
                   condition.getType() == esConditionType::TotalEtEM ||
                   condition.getType() == esConditionType::TotalHt ||
                   condition.getType() == esConditionType::MissingEt ||
                   condition.getType() == esConditionType::MissingHt ||
                   condition.getType() == esConditionType::MissingEtHF ||
                   condition.getType() == esConditionType::TowerCount ||
                   condition.getType() == esConditionType::MinBiasHFP0 ||
                   condition.getType() == esConditionType::MinBiasHFM0 ||
                   condition.getType() == esConditionType::MinBiasHFP1 ||
                   condition.getType() == esConditionType::MinBiasHFM1 ||
                   condition.getType() == esConditionType::AsymmetryEt ||
                   condition.getType() == esConditionType::AsymmetryHt ||
                   condition.getType() == esConditionType::AsymmetryEtHF ||
                   condition.getType() == esConditionType::AsymmetryHtHF ||
                   condition.getType() == esConditionType::Centrality0 ||
                   condition.getType() == esConditionType::Centrality1 ||
                   condition.getType() == esConditionType::Centrality2 ||
                   condition.getType() == esConditionType::Centrality3 ||
                   condition.getType() == esConditionType::Centrality4 ||
                   condition.getType() == esConditionType::Centrality5 ||
                   condition.getType() == esConditionType::Centrality6 ||
                   condition.getType() == esConditionType::Centrality7) {
          parseEnergySum(condition, chipNr, false);

          // parse ZDC Energy Sums (NOTE: HI trigger objects are treated as energy sums or counters)
        } else if (condition.getType() == esConditionType::ZDCPlus ||
                   condition.getType() == esConditionType::ZDCMinus) {
          parseEnergySumZdc(condition, chipNr, false);

          //parse AXOL1TL
        } else if (condition.getType() == esConditionType::AnomalyDetectionTrigger) {
          parseAXOL1TL(condition, chipNr);

          //parse Muons
        } else if (condition.getType() == esConditionType::SingleMuon ||
                   condition.getType() == esConditionType::DoubleMuon ||
                   condition.getType() == esConditionType::TripleMuon ||
                   condition.getType() == esConditionType::QuadMuon) {
          parseMuon(condition, chipNr, false);

        } else if (condition.getType() == esConditionType::MuonShower0 ||
                   condition.getType() == esConditionType::MuonShower1 ||
                   condition.getType() == esConditionType::MuonShower2 ||
                   condition.getType() == esConditionType::MuonShowerOutOfTime0 ||
                   condition.getType() == esConditionType::MuonShowerOutOfTime1) {
          parseMuonShower(condition, chipNr, false);

          //parse Correlation Conditions
        } else if (condition.getType() == esConditionType::MuonMuonCorrelation ||
                   condition.getType() == esConditionType::MuonEsumCorrelation ||
                   condition.getType() == esConditionType::CaloMuonCorrelation ||
                   condition.getType() == esConditionType::CaloCaloCorrelation ||
                   condition.getType() == esConditionType::CaloEsumCorrelation ||
                   condition.getType() == esConditionType::InvariantMass ||
                   condition.getType() == esConditionType::InvariantMassDeltaR ||
                   condition.getType() == esConditionType::TransverseMass ||
                   condition.getType() == esConditionType::InvariantMassUpt) {  // Added for displaced muons
          parseCorrelation(condition, chipNr);

          //parse three-body Correlation Conditions
        } else if (condition.getType() == esConditionType::InvariantMass3) {
          parseCorrelationThreeBody(condition, chipNr);

          //parse Externals
        } else if (condition.getType() == esConditionType::Externals) {
          parseExternal(condition, chipNr);

          //parse CorrelationWithOverlapRemoval
        } else if (condition.getType() == esConditionType::CaloCaloCorrelationOvRm ||
                   condition.getType() == esConditionType::InvariantMassOvRm ||
                   condition.getType() == esConditionType::TransverseMassOvRm ||
                   condition.getType() == esConditionType::DoubleJetOvRm ||
                   condition.getType() == esConditionType::DoubleTauOvRm) {
          parseCorrelationWithOverlapRemoval(condition, chipNr);

        } else if (condition.getType() == esConditionType::SingleEgammaOvRm ||
                   condition.getType() == esConditionType::DoubleEgammaOvRm ||
                   condition.getType() == esConditionType::TripleEgammaOvRm ||
                   condition.getType() == esConditionType::QuadEgammaOvRm ||
                   condition.getType() == esConditionType::SingleTauOvRm ||
                   condition.getType() == esConditionType::TripleTauOvRm ||
                   condition.getType() == esConditionType::QuadTauOvRm ||
                   condition.getType() == esConditionType::SingleJetOvRm ||
                   condition.getType() == esConditionType::TripleJetOvRm ||
                   condition.getType() == esConditionType::QuadJetOvRm) {
          edm::LogError("TriggerMenuParser") << std::endl
                                             << "\n SingleEgammaOvRm"
                                             << "\n DoubleEgammaOvRm"
                                             << "\n TripleEgammaOvRm"
                                             << "\n QuadEgammaOvRm"
                                             << "\n SingleTauOvRm"
                                             << "\n TripleTauOvRm"
                                             << "\n QuadTauOvRm"
                                             << "\n SingleJetOvRm"
                                             << "\n TripleJetOvRm"
                                             << "\n QuadJetOvRm"
                                             << "\n The above conditions types OvRm are not implemented yet in the "
                                                "parser. Please remove alogrithms that "
                                                "use this type of condtion from L1T Menu!"
                                             << std::endl;
        }

      }  //if condition is a new one
    }    //loop over conditions
  }      //loop over algorithms

  return;
}

//

void l1t::TriggerMenuParser::setGtTriggerMenuInterfaceDate(const std::string& val) { m_triggerMenuInterfaceDate = val; }

void l1t::TriggerMenuParser::setGtTriggerMenuInterfaceAuthor(const std::string& val) {
  m_triggerMenuInterfaceAuthor = val;
}

void l1t::TriggerMenuParser::setGtTriggerMenuInterfaceDescription(const std::string& val) {
  m_triggerMenuInterfaceDescription = val;
}

void l1t::TriggerMenuParser::setGtTriggerMenuDate(const std::string& val) { m_triggerMenuDate = val; }

void l1t::TriggerMenuParser::setGtTriggerMenuAuthor(const std::string& val) { m_triggerMenuAuthor = val; }

void l1t::TriggerMenuParser::setGtTriggerMenuDescription(const std::string& val) { m_triggerMenuDescription = val; }

void l1t::TriggerMenuParser::setGtAlgorithmImplementation(const std::string& val) { m_algorithmImplementation = val; }

// methods for conditions and algorithms

// clearMaps - delete all conditions and algorithms in
// the maps and clear the maps.
void l1t::TriggerMenuParser::clearMaps() {
  // loop over condition maps (one map per condition chip)
  // then loop over conditions in the map
  for (std::vector<ConditionMap>::iterator itCondOnChip = m_conditionMap.begin(); itCondOnChip != m_conditionMap.end();
       itCondOnChip++) {
    // the conditions in the maps are deleted in L1uGtTriggerMenu, not here

    itCondOnChip->clear();
  }

  // the algorithms in the maps are deleted in L1uGtTriggerMenu, not here
  m_algorithmMap.clear();
}

// insertConditionIntoMap - safe insert of condition into condition map.
// if the condition name already exists, do not insert it and return false
bool l1t::TriggerMenuParser::insertConditionIntoMap(GlobalCondition& cond, const int chipNr) {
  std::string cName = cond.condName();
  LogTrace("TriggerMenuParser") << "    Trying to insert condition \"" << cName << "\" in the condition map."
                                << std::endl;

  // no condition name has to appear twice!
  if ((m_conditionMap[chipNr]).count(cName) != 0) {
    LogTrace("TriggerMenuParser") << "      Condition " << cName << " already exists - not inserted!" << std::endl;
    return false;
  }

  (m_conditionMap[chipNr])[cName] = &cond;
  LogTrace("TriggerMenuParser") << "      OK - condition inserted!" << std::endl;

  return true;
}

// insert an algorithm into algorithm map
bool l1t::TriggerMenuParser::insertAlgorithmIntoMap(const GlobalAlgorithm& alg) {
  std::string algName = alg.algoName();
  const std::string& algAlias = alg.algoAlias();
  //LogTrace("TriggerMenuParser")
  //<< "    Trying to insert algorithm \"" << algName << "\" in the algorithm map." ;

  // no algorithm name has to appear twice!
  if (m_algorithmMap.count(algName) != 0) {
    LogTrace("TriggerMenuParser") << "      Algorithm \"" << algName
                                  << "\"already exists in the algorithm map- not inserted!" << std::endl;
    return false;
  }

  if (m_algorithmAliasMap.count(algAlias) != 0) {
    LogTrace("TriggerMenuParser") << "      Algorithm alias \"" << algAlias
                                  << "\"already exists in the algorithm alias map- not inserted!" << std::endl;
    return false;
  }

  // bit number less than zero or greater than maximum number of algorithms
  int bitNumber = alg.algoBitNumber();
  if ((bitNumber < 0) || (bitNumber >= static_cast<int>(m_numberPhysTriggers))) {
    LogTrace("TriggerMenuParser") << "      Bit number " << bitNumber << " outside allowed range [0, "
                                  << m_numberPhysTriggers << ") - algorithm not inserted!" << std::endl;
    return false;
  }

  // maximum number of algorithms
  if (m_algorithmMap.size() >= m_numberPhysTriggers) {
    LogTrace("TriggerMenuParser") << "      More than maximum allowed " << m_numberPhysTriggers
                                  << " algorithms in the algorithm map - not inserted!" << std::endl;
    return false;
  }

  // chip number outside allowed values
  int chipNr = alg.algoChipNumber(
      static_cast<int>(m_numberConditionChips), static_cast<int>(m_pinsOnConditionChip), m_orderConditionChip);

  if ((chipNr < 0) || (chipNr > static_cast<int>(m_numberConditionChips))) {
    LogTrace("TriggerMenuParser") << "      Chip number " << chipNr << " outside allowed range [0, "
                                  << m_numberConditionChips << ") - algorithm not inserted!" << std::endl;
    return false;
  }

  // output pin outside allowed values
  int outputPin = alg.algoOutputPin(
      static_cast<int>(m_numberConditionChips), static_cast<int>(m_pinsOnConditionChip), m_orderConditionChip);

  if ((outputPin < 0) || (outputPin > static_cast<int>(m_pinsOnConditionChip))) {
    LogTrace("TriggerMenuParser") << "      Output pin " << outputPin << " outside allowed range [0, "
                                  << m_pinsOnConditionChip << "] - algorithm not inserted!" << std::endl;
    return false;
  }

  // no two algorithms on the same chip can have the same output pin
  for (CItAlgo itAlgo = m_algorithmMap.begin(); itAlgo != m_algorithmMap.end(); itAlgo++) {
    int iPin = (itAlgo->second)
                   .algoOutputPin(static_cast<int>(m_numberConditionChips),
                                  static_cast<int>(m_pinsOnConditionChip),
                                  m_orderConditionChip);
    std::string iName = itAlgo->first;
    int iChip = (itAlgo->second)
                    .algoChipNumber(static_cast<int>(m_numberConditionChips),
                                    static_cast<int>(m_pinsOnConditionChip),
                                    m_orderConditionChip);

    if ((outputPin == iPin) && (chipNr == iChip)) {
      LogTrace("TriggerMenuParser") << "      Output pin " << outputPin << " is the same as for algorithm " << iName
                                    << "\n      from the same chip number " << chipNr << " - algorithm not inserted!"
                                    << std::endl;
      return false;
    }
  }

  // insert algorithm
  m_algorithmMap[algName] = alg;
  m_algorithmAliasMap[algAlias] = alg;

  //LogTrace("TriggerMenuParser")
  //<< "      OK - algorithm inserted!"
  //<< std::endl;

  return true;
}

template <typename T>
std::string l1t::TriggerMenuParser::l1t2string(T data) {
  std::stringstream ss;
  ss << data;
  return ss.str();
}
int l1t::TriggerMenuParser::l1tstr2int(const std::string data) {
  std::stringstream ss;
  ss << data;
  int value;
  ss >> value;
  return value;
}

/**
 * parseScales Parse Et, Eta, and Phi Scales
 *
 * @return "true" if succeeded, "false" if an error occurred.
 *
 */

bool l1t::TriggerMenuParser::parseScales(std::map<std::string, tmeventsetup::esScale> scaleMap) {
  using namespace tmeventsetup;

  //  Setup ScaleParameter to hold information from parsing
  GlobalScales::ScaleParameters muScales;
  GlobalScales::ScaleParameters egScales;
  GlobalScales::ScaleParameters tauScales;
  GlobalScales::ScaleParameters jetScales;
  GlobalScales::ScaleParameters ettScales;
  GlobalScales::ScaleParameters ettEmScales;
  GlobalScales::ScaleParameters etmScales;
  GlobalScales::ScaleParameters etmHfScales;
  GlobalScales::ScaleParameters httScales;
  GlobalScales::ScaleParameters htmScales;
  GlobalScales::ScaleParameters zdcScales;

  // Start by parsing the Scale Map
  for (std::map<std::string, tmeventsetup::esScale>::const_iterator cit = scaleMap.begin(); cit != scaleMap.end();
       cit++) {
    const tmeventsetup::esScale& scale = cit->second;

    GlobalScales::ScaleParameters* scaleParam;
    if (scale.getObjectType() == esObjectType::Muon)
      scaleParam = &muScales;
    else if (scale.getObjectType() == esObjectType::Egamma)
      scaleParam = &egScales;
    else if (scale.getObjectType() == esObjectType::Tau)
      scaleParam = &tauScales;
    else if (scale.getObjectType() == esObjectType::Jet)
      scaleParam = &jetScales;
    else if (scale.getObjectType() == esObjectType::ETT)
      scaleParam = &ettScales;
    else if (scale.getObjectType() == esObjectType::ETTEM)
      scaleParam = &ettEmScales;
    else if (scale.getObjectType() == esObjectType::ETM)
      scaleParam = &etmScales;
    else if (scale.getObjectType() == esObjectType::ETMHF)
      scaleParam = &etmHfScales;
    else if (scale.getObjectType() == esObjectType::HTT)
      scaleParam = &httScales;
    else if (scale.getObjectType() == esObjectType::HTM)
      scaleParam = &htmScales;
    else if (scale.getObjectType() == esObjectType::ZDCP || scale.getObjectType() == esObjectType::ZDCM)
      scaleParam = &zdcScales;
    else
      scaleParam = nullptr;

    if (scaleParam != nullptr) {
      switch (scale.getScaleType()) {
        case esScaleType::EtScale: {
          scaleParam->etMin = scale.getMinimum();
          scaleParam->etMax = scale.getMaximum();
          scaleParam->etStep = scale.getStep();

          //Get bin edges
          const std::vector<tmeventsetup::esBin>& binsV = scale.getBins();
          for (unsigned int i = 0; i < binsV.size(); i++) {
            const tmeventsetup::esBin& bin = binsV.at(i);
            std::pair<double, double> binLimits(bin.minimum, bin.maximum);
            scaleParam->etBins.push_back(binLimits);
          }

          // If this is an energy sum fill dummy values for eta and phi
          // There are no scales for these in the XML so the other case statements will not be seen....do it here.
          if (scale.getObjectType() == esObjectType::ETT || scale.getObjectType() == esObjectType::HTT ||
              scale.getObjectType() == esObjectType::ETM || scale.getObjectType() == esObjectType::HTM ||
              scale.getObjectType() == esObjectType::ETTEM || scale.getObjectType() == esObjectType::ETMHF) {
            scaleParam->etaMin = -1.;
            scaleParam->etaMax = -1.;
            scaleParam->etaStep = -1.;
            if (scale.getObjectType() == esObjectType::ETT || scale.getObjectType() == esObjectType::HTT ||
                scale.getObjectType() == esObjectType::ETTEM) {
              //		   if(scale.getObjectType() == esObjectType::ETT || scale.getObjectType() == esObjectType::HTT) {
              scaleParam->phiMin = -1.;
              scaleParam->phiMax = -1.;
              scaleParam->phiStep = -1.;
            }
          }
        } break;
        case esScaleType::UnconstrainedPtScale: {  // Added for displaced muons
          scaleParam->uptMin = scale.getMinimum();
          scaleParam->uptMax = scale.getMaximum();
          scaleParam->uptStep = scale.getStep();

          //Get bin edges
          const std::vector<tmeventsetup::esBin>& binsV = scale.getBins();
          for (unsigned int i = 0; i < binsV.size(); i++) {
            const tmeventsetup::esBin& bin = binsV.at(i);
            std::pair<double, double> binLimits(bin.minimum, bin.maximum);
            scaleParam->uptBins.push_back(binLimits);
          }
        } break;
        case esScaleType::EtaScale: {
          scaleParam->etaMin = scale.getMinimum();
          scaleParam->etaMax = scale.getMaximum();
          scaleParam->etaStep = scale.getStep();

          //Get bin edges
          const std::vector<tmeventsetup::esBin>& binsV = scale.getBins();
          scaleParam->etaBins.resize(pow(2, scale.getNbits()));
          for (unsigned int i = 0; i < binsV.size(); i++) {
            const tmeventsetup::esBin& bin = binsV.at(i);
            std::pair<double, double> binLimits(bin.minimum, bin.maximum);
            scaleParam->etaBins.at(bin.hw_index) = binLimits;
          }
        } break;
        case esScaleType::PhiScale: {
          scaleParam->phiMin = scale.getMinimum();
          scaleParam->phiMax = scale.getMaximum();
          scaleParam->phiStep = scale.getStep();

          //Get bin edges
          const std::vector<tmeventsetup::esBin>& binsV = scale.getBins();
          scaleParam->phiBins.resize(pow(2, scale.getNbits()));
          for (unsigned int i = 0; i < binsV.size(); i++) {
            const tmeventsetup::esBin& bin = binsV.at(i);
            std::pair<double, double> binLimits(bin.minimum, bin.maximum);
            scaleParam->phiBins.at(bin.hw_index) = binLimits;
          }
        } break;
        default:

          break;
      }  //end switch
    }    //end valid scale
  }      //end loop over scaleMap

  // put the ScaleParameters into the class
  m_gtScales.setMuonScales(muScales);
  m_gtScales.setEGScales(egScales);
  m_gtScales.setTauScales(tauScales);
  m_gtScales.setJetScales(jetScales);
  m_gtScales.setETTScales(ettScales);
  m_gtScales.setETTEmScales(ettEmScales);
  m_gtScales.setETMScales(etmScales);
  m_gtScales.setETMHfScales(etmHfScales);
  m_gtScales.setHTTScales(httScales);
  m_gtScales.setHTMScales(htmScales);
  m_gtScales.setHTMScales(zdcScales);

  // Setup the LUT for the Scale Conversions
  bool hasPrecision = false;
  std::map<std::string, unsigned int> precisions;
  getPrecisions(precisions, scaleMap);
  for (std::map<std::string, unsigned int>::const_iterator cit = precisions.begin(); cit != precisions.end(); cit++) {
    hasPrecision = true;
  }

  if (hasPrecision) {
    //Start with the Cal - Muon Eta LUTS
    //----------------------------------
    parseCalMuEta_LUTS(scaleMap, "EG", "MU");
    parseCalMuEta_LUTS(scaleMap, "JET", "MU");
    parseCalMuEta_LUTS(scaleMap, "TAU", "MU");

    //Now the Cal - Muon Phi LUTS
    //-------------------------------------
    parseCalMuPhi_LUTS(scaleMap, "EG", "MU");
    parseCalMuPhi_LUTS(scaleMap, "JET", "MU");
    parseCalMuPhi_LUTS(scaleMap, "TAU", "MU");
    parseCalMuPhi_LUTS(scaleMap, "HTM", "MU");
    parseCalMuPhi_LUTS(scaleMap, "ETM", "MU");
    parseCalMuPhi_LUTS(scaleMap, "ETMHF", "MU");

    // Now the Pt LUTs  (??? more combinations needed ??)
    // ---------------
    parsePt_LUTS(scaleMap, "Mass", "EG", precisions["PRECISION-EG-MU-MassPt"]);
    parsePt_LUTS(scaleMap, "Mass", "MU", precisions["PRECISION-EG-MU-MassPt"]);
    parseUpt_LUTS(scaleMap, "Mass", "MU", precisions["PRECISION-EG-MU-MassPt"]);  // Added for displaced muons
    parsePt_LUTS(scaleMap, "Mass", "JET", precisions["PRECISION-EG-JET-MassPt"]);
    parsePt_LUTS(scaleMap, "Mass", "TAU", precisions["PRECISION-EG-TAU-MassPt"]);
    parsePt_LUTS(scaleMap, "Mass", "ETM", precisions["PRECISION-EG-ETM-MassPt"]);
    parsePt_LUTS(scaleMap, "Mass", "ETMHF", precisions["PRECISION-EG-ETMHF-MassPt"]);
    parsePt_LUTS(scaleMap, "Mass", "HTM", precisions["PRECISION-EG-HTM-MassPt"]);

    // Now the Pt LUTs  for TBPT calculation (??? CCLA following what was done for MASS pt LUTs for now ??)
    // ---------------
    parsePt_LUTS(scaleMap, "TwoBody", "EG", precisions["PRECISION-EG-MU-TwoBodyPt"]);
    parsePt_LUTS(scaleMap, "TwoBody", "MU", precisions["PRECISION-EG-MU-TwoBodyPt"]);
    parsePt_LUTS(scaleMap, "TwoBody", "JET", precisions["PRECISION-EG-JET-TwoBodyPt"]);
    parsePt_LUTS(scaleMap, "TwoBody", "TAU", precisions["PRECISION-EG-TAU-TwoBodyPt"]);
    parsePt_LUTS(scaleMap, "TwoBody", "ETM", precisions["PRECISION-EG-ETM-TwoBodyPt"]);
    parsePt_LUTS(scaleMap, "TwoBody", "ETMHF", precisions["PRECISION-EG-ETMHF-TwoBodyPt"]);
    parsePt_LUTS(scaleMap, "TwoBody", "HTM", precisions["PRECISION-EG-HTM-TwoBodyPt"]);

    // Now the Delta Eta/Cosh LUTs (must be done in groups)
    // ----------------------------------------------------
    parseDeltaEta_Cosh_LUTS(
        scaleMap, "EG", "EG", precisions["PRECISION-EG-EG-Delta"], precisions["PRECISION-EG-EG-Math"]);
    parseDeltaEta_Cosh_LUTS(
        scaleMap, "EG", "JET", precisions["PRECISION-EG-JET-Delta"], precisions["PRECISION-EG-JET-Math"]);
    parseDeltaEta_Cosh_LUTS(
        scaleMap, "EG", "TAU", precisions["PRECISION-EG-TAU-Delta"], precisions["PRECISION-EG-TAU-Math"]);
    parseDeltaEta_Cosh_LUTS(
        scaleMap, "EG", "MU", precisions["PRECISION-EG-MU-Delta"], precisions["PRECISION-EG-MU-Math"]);

    parseDeltaEta_Cosh_LUTS(
        scaleMap, "JET", "JET", precisions["PRECISION-JET-JET-Delta"], precisions["PRECISION-JET-JET-Math"]);
    parseDeltaEta_Cosh_LUTS(
        scaleMap, "JET", "TAU", precisions["PRECISION-JET-TAU-Delta"], precisions["PRECISION-JET-TAU-Math"]);
    parseDeltaEta_Cosh_LUTS(
        scaleMap, "JET", "MU", precisions["PRECISION-JET-MU-Delta"], precisions["PRECISION-JET-MU-Math"]);

    parseDeltaEta_Cosh_LUTS(
        scaleMap, "TAU", "TAU", precisions["PRECISION-TAU-TAU-Delta"], precisions["PRECISION-TAU-TAU-Math"]);
    parseDeltaEta_Cosh_LUTS(
        scaleMap, "TAU", "MU", precisions["PRECISION-TAU-MU-Delta"], precisions["PRECISION-TAU-MU-Math"]);

    parseDeltaEta_Cosh_LUTS(
        scaleMap, "MU", "MU", precisions["PRECISION-MU-MU-Delta"], precisions["PRECISION-MU-MU-Math"]);

    // Now the Delta Phi/Cos LUTs (must be done in groups)
    // ----------------------------------------------------
    parseDeltaPhi_Cos_LUTS(
        scaleMap, "EG", "EG", precisions["PRECISION-EG-EG-Delta"], precisions["PRECISION-EG-EG-Math"]);
    parseDeltaPhi_Cos_LUTS(
        scaleMap, "EG", "JET", precisions["PRECISION-EG-JET-Delta"], precisions["PRECISION-EG-JET-Math"]);
    parseDeltaPhi_Cos_LUTS(
        scaleMap, "EG", "TAU", precisions["PRECISION-EG-TAU-Delta"], precisions["PRECISION-EG-TAU-Math"]);
    parseDeltaPhi_Cos_LUTS(
        scaleMap, "EG", "ETM", precisions["PRECISION-EG-ETM-Delta"], precisions["PRECISION-EG-ETM-Math"]);
    parseDeltaPhi_Cos_LUTS(
        scaleMap, "EG", "ETMHF", precisions["PRECISION-EG-ETMHF-Delta"], precisions["PRECISION-EG-ETMHF-Math"]);
    parseDeltaPhi_Cos_LUTS(
        scaleMap, "EG", "HTM", precisions["PRECISION-EG-HTM-Delta"], precisions["PRECISION-EG-HTM-Math"]);
    parseDeltaPhi_Cos_LUTS(
        scaleMap, "EG", "MU", precisions["PRECISION-EG-MU-Delta"], precisions["PRECISION-EG-MU-Math"]);

    parseDeltaPhi_Cos_LUTS(
        scaleMap, "JET", "JET", precisions["PRECISION-JET-JET-Delta"], precisions["PRECISION-JET-JET-Math"]);
    parseDeltaPhi_Cos_LUTS(
        scaleMap, "JET", "TAU", precisions["PRECISION-JET-TAU-Delta"], precisions["PRECISION-JET-TAU-Math"]);
    parseDeltaPhi_Cos_LUTS(
        scaleMap, "JET", "ETM", precisions["PRECISION-JET-ETM-Delta"], precisions["PRECISION-JET-ETM-Math"]);
    parseDeltaPhi_Cos_LUTS(
        scaleMap, "JET", "ETMHF", precisions["PRECISION-JET-ETMHF-Delta"], precisions["PRECISION-JET-ETMHF-Math"]);
    parseDeltaPhi_Cos_LUTS(
        scaleMap, "JET", "HTM", precisions["PRECISION-JET-HTM-Delta"], precisions["PRECISION-JET-HTM-Math"]);
    parseDeltaPhi_Cos_LUTS(
        scaleMap, "JET", "MU", precisions["PRECISION-JET-MU-Delta"], precisions["PRECISION-JET-MU-Math"]);

    parseDeltaPhi_Cos_LUTS(
        scaleMap, "TAU", "TAU", precisions["PRECISION-TAU-TAU-Delta"], precisions["PRECISION-TAU-TAU-Math"]);
    parseDeltaPhi_Cos_LUTS(
        scaleMap, "TAU", "ETM", precisions["PRECISION-TAU-ETM-Delta"], precisions["PRECISION-TAU-ETM-Math"]);
    parseDeltaPhi_Cos_LUTS(
        scaleMap, "TAU", "ETMHF", precisions["PRECISION-TAU-ETMHF-Delta"], precisions["PRECISION-TAU-ETMHF-Math"]);
    parseDeltaPhi_Cos_LUTS(
        scaleMap, "TAU", "HTM", precisions["PRECISION-TAU-HTM-Delta"], precisions["PRECISION-TAU-HTM-Math"]);
    parseDeltaPhi_Cos_LUTS(
        scaleMap, "TAU", "MU", precisions["PRECISION-TAU-MU-Delta"], precisions["PRECISION-TAU-MU-Math"]);

    parseDeltaPhi_Cos_LUTS(
        scaleMap, "MU", "ETM", precisions["PRECISION-MU-ETM-Delta"], precisions["PRECISION-MU-ETM-Math"]);
    parseDeltaPhi_Cos_LUTS(
        scaleMap, "MU", "ETMHF", precisions["PRECISION-MU-ETMHF-Delta"], precisions["PRECISION-MU-ETMHF-Math"]);
    parseDeltaPhi_Cos_LUTS(
        scaleMap, "MU", "HTM", precisions["PRECISION-MU-HTM-Delta"], precisions["PRECISION-MU-HTM-Math"]);
    parseDeltaPhi_Cos_LUTS(
        scaleMap, "MU", "MU", precisions["PRECISION-MU-MU-Delta"], precisions["PRECISION-MU-MU-Math"]);

    parsePhi_Trig_LUTS(scaleMap, "EG", l1t::COS, precisions["PRECISION-EG-EG-Math"]);
    parsePhi_Trig_LUTS(scaleMap, "JET", l1t::COS, precisions["PRECISION-JET-JET-Math"]);
    parsePhi_Trig_LUTS(scaleMap, "TAU", l1t::COS, precisions["PRECISION-TAU-TAU-Math"]);
    parsePhi_Trig_LUTS(scaleMap, "MU", l1t::COS, precisions["PRECISION-MU-MU-Math"]);

    parsePhi_Trig_LUTS(scaleMap, "EG", l1t::SIN, precisions["PRECISION-EG-EG-Math"]);
    parsePhi_Trig_LUTS(scaleMap, "JET", l1t::SIN, precisions["PRECISION-JET-JET-Math"]);
    parsePhi_Trig_LUTS(scaleMap, "TAU", l1t::SIN, precisions["PRECISION-TAU-TAU-Math"]);
    parsePhi_Trig_LUTS(scaleMap, "MU", l1t::SIN, precisions["PRECISION-MU-MU-Math"]);

    //CCLA
    //m_gtScales.dumpAllLUTs(std::cout);
    //m_gtScales.print(std::cout);
  }

  return true;
}

void l1t::TriggerMenuParser::parseCalMuEta_LUTS(std::map<std::string, tmeventsetup::esScale> scaleMap,
                                                std::string obj1,
                                                std::string obj2) {
  using namespace tmeventsetup;

  // First Delta Eta for this set
  std::string scLabel1 = obj1;
  scLabel1 += "-ETA";
  std::string scLabel2 = obj2;
  scLabel2 += "-ETA";

  //This LUT does not exist in L1 Menu file, don't fill it
  if (scaleMap.find(scLabel1) == scaleMap.end() || scaleMap.find(scLabel2) == scaleMap.end())
    return;

  const tmeventsetup::esScale* scale1 = &scaleMap.find(scLabel1)->second;
  const tmeventsetup::esScale* scale2 = &scaleMap.find(scLabel2)->second;

  std::vector<long long> lut_cal_2_mu_eta;
  getCaloMuonEtaConversionLut(lut_cal_2_mu_eta, scale1, scale2);

  std::string lutName = obj1;
  lutName += "-";
  lutName += obj2;
  m_gtScales.setLUT_CalMuEta(lutName, lut_cal_2_mu_eta);
}

void l1t::TriggerMenuParser::parseCalMuPhi_LUTS(std::map<std::string, tmeventsetup::esScale> scaleMap,
                                                std::string obj1,
                                                std::string obj2) {
  using namespace tmeventsetup;

  // First Delta Eta for this set
  std::string scLabel1 = obj1;
  scLabel1 += "-PHI";
  std::string scLabel2 = obj2;
  scLabel2 += "-PHI";

  //This LUT does not exist in L1 Menu file, don't fill it
  if (scaleMap.find(scLabel1) == scaleMap.end() || scaleMap.find(scLabel2) == scaleMap.end())
    return;

  const tmeventsetup::esScale* scale1 = &scaleMap.find(scLabel1)->second;
  const tmeventsetup::esScale* scale2 = &scaleMap.find(scLabel2)->second;

  std::vector<long long> lut_cal_2_mu_phi;
  getCaloMuonPhiConversionLut(lut_cal_2_mu_phi, scale1, scale2);

  std::string lutName = obj1;
  lutName += "-";
  lutName += obj2;
  m_gtScales.setLUT_CalMuPhi(lutName, lut_cal_2_mu_phi);
}

void l1t::TriggerMenuParser::parsePt_LUTS(std::map<std::string, tmeventsetup::esScale> scaleMap,
                                          std::string lutpfx,
                                          std::string obj1,
                                          unsigned int prec) {
  using namespace tmeventsetup;

  // First Delta Eta for this set
  std::string scLabel1 = obj1;
  scLabel1 += "-ET";

  //This LUT does not exist in L1 Menu file, don't fill it
  if (scaleMap.find(scLabel1) == scaleMap.end())
    return;

  const tmeventsetup::esScale* scale1 = &scaleMap.find(scLabel1)->second;

  std::vector<long long> lut_pt;
  getLut(lut_pt, scale1, prec);

  m_gtScales.setLUT_Pt(lutpfx + "_" + scLabel1, lut_pt, prec);
}

// Added for displaced muons
void l1t::TriggerMenuParser::parseUpt_LUTS(std::map<std::string, tmeventsetup::esScale> scaleMap,
                                           std::string lutpfx,
                                           std::string obj1,
                                           unsigned int prec) {
  using namespace tmeventsetup;

  // First Delta Eta for this set
  std::string scLabel1 = obj1;
  scLabel1 += "-UPT";

  //This LUT does not exist in L1 Menu file, don't fill it
  if (scaleMap.find(scLabel1) == scaleMap.end())
    return;

  const tmeventsetup::esScale* scale1 = &scaleMap.find(scLabel1)->second;

  std::vector<long long> lut_pt;
  getLut(lut_pt, scale1, prec);

  m_gtScales.setLUT_Upt(lutpfx + "_" + scLabel1, lut_pt, prec);
}

void l1t::TriggerMenuParser::parseDeltaEta_Cosh_LUTS(std::map<std::string, tmeventsetup::esScale> scaleMap,
                                                     std::string obj1,
                                                     std::string obj2,
                                                     unsigned int prec1,
                                                     unsigned int prec2) {
  using namespace tmeventsetup;

  // First Delta Eta for this set
  std::string scLabel1 = obj1;
  scLabel1 += "-ETA";
  std::string scLabel2 = obj2;
  scLabel2 += "-ETA";

  //This LUT does not exist in L1 Menu file, don't fill it
  if (scaleMap.find(scLabel1) == scaleMap.end() || scaleMap.find(scLabel2) == scaleMap.end())
    return;

  const tmeventsetup::esScale* scale1 = &scaleMap.find(scLabel1)->second;
  const tmeventsetup::esScale* scale2 = &scaleMap.find(scLabel2)->second;
  std::vector<double> val_delta_eta;
  std::vector<long long> lut_delta_eta;
  size_t n = getDeltaVector(val_delta_eta, scale1, scale2);
  setLut(lut_delta_eta, val_delta_eta, prec1);
  std::string lutName = obj1;
  lutName += "-";
  lutName += obj2;
  m_gtScales.setLUT_DeltaEta(lutName, lut_delta_eta, prec1);

  // Second Get the Cosh for this delta Eta Set
  std::vector<long long> lut_cosh;
  applyCosh(val_delta_eta, n);
  setLut(lut_cosh, val_delta_eta, prec2);
  m_gtScales.setLUT_Cosh(lutName, lut_cosh, prec2);
}

void l1t::TriggerMenuParser::parseDeltaPhi_Cos_LUTS(const std::map<std::string, tmeventsetup::esScale>& scaleMap,
                                                    const std::string& obj1,
                                                    const std::string& obj2,
                                                    unsigned int prec1,
                                                    unsigned int prec2) {
  using namespace tmeventsetup;

  // First Delta phi for this set
  std::string scLabel1 = obj1;
  scLabel1 += "-PHI";
  std::string scLabel2 = obj2;
  scLabel2 += "-PHI";

  //This LUT does not exist in L1 Menu file, don't fill it
  if (scaleMap.find(scLabel1) == scaleMap.end() || scaleMap.find(scLabel2) == scaleMap.end())
    return;

  const tmeventsetup::esScale* scale1 = &scaleMap.find(scLabel1)->second;
  const tmeventsetup::esScale* scale2 = &scaleMap.find(scLabel2)->second;
  std::vector<double> val_delta_phi;
  std::vector<long long> lut_delta_phi;
  size_t n = getDeltaVector(val_delta_phi, scale1, scale2);
  setLut(lut_delta_phi, val_delta_phi, prec1);
  std::string lutName = obj1;
  lutName += "-";
  lutName += obj2;
  m_gtScales.setLUT_DeltaPhi(lutName, lut_delta_phi, prec1);

  // Second Get the Cosh for this delta phi Set
  std::vector<long long> lut_cos;
  applyCos(val_delta_phi, n);
  setLut(lut_cos, val_delta_phi, prec2);
  m_gtScales.setLUT_Cos(lutName, lut_cos, prec2);
}

void l1t::TriggerMenuParser::parsePhi_Trig_LUTS(const std::map<std::string, tmeventsetup::esScale>& scaleMap,
                                                const std::string& obj,
                                                l1t::TrigFunc_t func,
                                                unsigned int prec) {
  using namespace tmeventsetup;

  std::string scLabel = obj;
  scLabel += "-PHI";

  //This LUT does not exist in L1 Menu file, don't fill it
  if (scaleMap.find(scLabel) == scaleMap.end())
    return;
  if (func != l1t::SIN and func != l1t::COS)
    return;

  const tmeventsetup::esScale* scale = &scaleMap.find(scLabel)->second;

  const double step = scale->getStep();
  const double range = scale->getMaximum() - scale->getMinimum();
  const size_t n = std::ceil(range / step);
  const size_t bitwidth = std::ceil(std::log10(n) / std::log10(2));

  std::vector<double> array(std::pow(2, bitwidth), 0);

  for (size_t ii = 0; ii < n; ii++) {
    array.at(ii) = step * ii;
  }

  const std::string& lutName = obj;
  std::vector<long long> lut;
  if (func == l1t::SIN) {
    applySin(array, n);
    setLut(lut, array, prec);
    m_gtScales.setLUT_Sin(lutName, lut, prec);
  } else if (func == l1t::COS) {
    applyCos(array, n);
    setLut(lut, array, prec);
    m_gtScales.setLUT_Cos(lutName, lut, prec);
  }
}

/**
 * parseMuon Parse a muon condition and insert an entry to the conditions map
 *
 * @param node The corresponding node.
 * @param name The name of the condition.
 * @param chipNr The number of the chip this condition is located.
 *
 * @return "true" if succeeded, "false" if an error occurred.
 *
 */

bool l1t::TriggerMenuParser::parseMuon(L1TUtmCondition condMu, unsigned int chipNr, const bool corrFlag) {
  using namespace tmeventsetup;
  // get condition, particle name (must be muon) and type name
  std::string condition = "muon";
  std::string particle = "muon";  //l1t2string( condMu.objectType() );
  std::string type = l1t2string(condMu.getType());
  std::string name = l1t2string(condMu.getName());
  int nrObj = -1;

  GtConditionType cType = l1t::TypeNull;

  if (condMu.getType() == esConditionType::SingleMuon) {
    type = "1_s";
    cType = l1t::Type1s;
    nrObj = 1;
  } else if (condMu.getType() == esConditionType::DoubleMuon) {
    type = "2_s";
    cType = l1t::Type2s;
    nrObj = 2;
  } else if (condMu.getType() == esConditionType::TripleMuon) {
    type = "3";
    cType = l1t::Type3s;
    nrObj = 3;
  } else if (condMu.getType() == esConditionType::QuadMuon) {
    type = "4";
    cType = l1t::Type4s;
    nrObj = 4;
  } else {
    edm::LogError("TriggerMenuParser") << "Wrong type for muon-condition (" << type << ")" << std::endl;
    return false;
  }

  if (nrObj < 0) {
    edm::LogError("TriggerMenuParser") << "Unknown type for muon-condition (" << type << ")"
                                       << "\nCan not determine number of trigger objects. " << std::endl;
    return false;
  }

  LogDebug("TriggerMenuParser") << "\n ****************************************** "
                                << "\n      parseMuon  "
                                << "\n condition = " << condition << "\n particle  = " << particle
                                << "\n type      = " << type << "\n name      = " << name << std::endl;

  //     // get values

  // temporary storage of the parameters
  std::vector<MuonTemplate::ObjectParameter> objParameter(nrObj);

  // Do we need this?
  MuonTemplate::CorrelationParameter corrParameter;

  // need at least two values for deltaPhi
  std::vector<uint64_t> tmpValues((nrObj > 2) ? nrObj : 2);
  tmpValues.reserve(nrObj);

  if (int(condMu.getObjects().size()) != nrObj) {
    edm::LogError("TriggerMenuParser") << " condMu objects: nrObj = " << nrObj
                                       << "condMu.getObjects().size() = " << condMu.getObjects().size() << std::endl;
    return false;
  }

  //  Look for cuts on the objects in the condition
  unsigned int chargeCorrelation = 1;
  const std::vector<L1TUtmCut>& cuts = condMu.getCuts();
  for (size_t jj = 0; jj < cuts.size(); jj++) {
    const L1TUtmCut& cut = cuts.at(jj);
    if (cut.getCutType() == esCutType::ChargeCorrelation) {
      if (cut.getData() == "ls")
        chargeCorrelation = 2;
      else if (cut.getData() == "os")
        chargeCorrelation = 4;
      else
        chargeCorrelation = 1;  //ignore correlation
    }
  }

  //set charge correlation parameter
  corrParameter.chargeCorrelation = chargeCorrelation;  //tmpValues[0];

  int cnt = 0;

  // BLW TO DO: These needs to the added to the object rather than the whole condition.
  int relativeBx = 0;
  bool gEq = false;

  // Loop over objects and extract the cuts on the objects
  const std::vector<L1TUtmObject>& objects = condMu.getObjects();
  for (size_t jj = 0; jj < objects.size(); jj++) {
    const L1TUtmObject& object = objects.at(jj);
    gEq = (object.getComparisonOperator() == esComparisonOperator::GE);

    //  BLW TO DO: This needs to be added to the Object Parameters
    relativeBx = object.getBxOffset();

    //  Loop over the cuts for this object
    int upperUnconstrainedPtInd = -1;
    int lowerUnconstrainedPtInd = 0;
    int upperImpactParameterInd = -1;
    int lowerImpactParameterInd = 0;
    int upperThresholdInd = -1;
    int lowerThresholdInd = 0;
    int upperIndexInd = -1;
    int lowerIndexInd = 0;
    int cntPhi = 0;
    unsigned int phiWindow1Lower = -1, phiWindow1Upper = -1, phiWindow2Lower = -1, phiWindow2Upper = -1;
    int isolationLUT = 0xF;        //default is to ignore unless specified.
    int impactParameterLUT = 0xF;  //default is to ignore unless specified
    int charge = -1;               //default value is to ignore unless specified
    int qualityLUT = 0xFFFF;       //default is to ignore unless specified.

    std::vector<MuonTemplate::Window> etaWindows;
    std::vector<MuonTemplate::Window> tfMuonIndexWindows;

    const std::vector<L1TUtmCut>& cuts = object.getCuts();
    for (size_t kk = 0; kk < cuts.size(); kk++) {
      const L1TUtmCut& cut = cuts.at(kk);

      switch (cut.getCutType()) {
        case esCutType::UnconstrainedPt:
          lowerUnconstrainedPtInd = cut.getMinimum().index;
          upperUnconstrainedPtInd = cut.getMaximum().index;
          break;

        case esCutType::ImpactParameter:
          lowerImpactParameterInd = cut.getMinimum().index;
          upperImpactParameterInd = cut.getMaximum().index;
          impactParameterLUT = l1tstr2int(cut.getData());
          break;

        case esCutType::Threshold:
          lowerThresholdInd = cut.getMinimum().index;
          upperThresholdInd = cut.getMaximum().index;
          break;

        case esCutType::Slice:
          lowerIndexInd = int(cut.getMinimum().value);
          upperIndexInd = int(cut.getMaximum().value);
          break;

        case esCutType::Eta: {
          if (etaWindows.size() < 5) {
            etaWindows.push_back({cut.getMinimum().index, cut.getMaximum().index});
          } else {
            edm::LogError("TriggerMenuParser")
                << "Too Many Eta Cuts for muon-condition (" << particle << ")" << std::endl;
            return false;
          }
        } break;

        case esCutType::Phi: {
          if (cntPhi == 0) {
            phiWindow1Lower = cut.getMinimum().index;
            phiWindow1Upper = cut.getMaximum().index;
          } else if (cntPhi == 1) {
            phiWindow2Lower = cut.getMinimum().index;
            phiWindow2Upper = cut.getMaximum().index;
          } else {
            edm::LogError("TriggerMenuParser")
                << "Too Many Phi Cuts for muon-condition (" << particle << ")" << std::endl;
            return false;
          }
          cntPhi++;

        } break;

        case esCutType::Charge:
          if (cut.getData() == "positive")
            charge = 0;
          else if (cut.getData() == "negative")
            charge = 1;
          else
            charge = -1;
          break;
        case esCutType::Quality:

          qualityLUT = l1tstr2int(cut.getData());

          break;
        case esCutType::Isolation: {
          isolationLUT = l1tstr2int(cut.getData());

        } break;

        case esCutType::Index: {
          tfMuonIndexWindows.push_back({cut.getMinimum().index, cut.getMaximum().index});
        } break;

        default:
          break;
      }  //end switch

    }  //end loop over cuts

    // Set the parameter cuts
    objParameter[cnt].unconstrainedPtHigh = upperUnconstrainedPtInd;
    objParameter[cnt].unconstrainedPtLow = lowerUnconstrainedPtInd;
    objParameter[cnt].impactParameterHigh = upperImpactParameterInd;
    objParameter[cnt].impactParameterLow = lowerImpactParameterInd;
    objParameter[cnt].impactParameterLUT = impactParameterLUT;

    objParameter[cnt].ptHighThreshold = upperThresholdInd;
    objParameter[cnt].ptLowThreshold = lowerThresholdInd;

    objParameter[cnt].indexHigh = upperIndexInd;
    objParameter[cnt].indexLow = lowerIndexInd;

    objParameter[cnt].etaWindows = etaWindows;

    objParameter[cnt].phiWindow1Lower = phiWindow1Lower;
    objParameter[cnt].phiWindow1Upper = phiWindow1Upper;
    objParameter[cnt].phiWindow2Lower = phiWindow2Lower;
    objParameter[cnt].phiWindow2Upper = phiWindow2Upper;

    // BLW TO DO: Do we need these anymore?  Drop them?
    objParameter[cnt].enableMip = false;   //tmpMip[i];
    objParameter[cnt].enableIso = false;   //tmpEnableIso[i];
    objParameter[cnt].requestIso = false;  //tmpRequestIso[i];

    objParameter[cnt].charge = charge;
    objParameter[cnt].qualityLUT = qualityLUT;
    objParameter[cnt].isolationLUT = isolationLUT;

    objParameter[cnt].tfMuonIndexWindows = tfMuonIndexWindows;

    cnt++;
  }  //end loop over objects

  // object types - all muons
  std::vector<GlobalObject> objType(nrObj, gtMu);

  // now create a new CondMuonition
  MuonTemplate muonCond(name);

  muonCond.setCondType(cType);
  muonCond.setObjectType(objType);
  muonCond.setCondGEq(gEq);
  muonCond.setCondChipNr(chipNr);
  muonCond.setCondRelativeBx(relativeBx);

  muonCond.setConditionParameter(objParameter, corrParameter);

  if (edm::isDebugEnabled()) {
    std::ostringstream myCoutStream;
    muonCond.print(myCoutStream);
    LogTrace("TriggerMenuParser") << myCoutStream.str() << "\n" << std::endl;
  }

  // insert condition into the map and into muon template vector
  if (!insertConditionIntoMap(muonCond, chipNr)) {
    edm::LogError("TriggerMenuParser") << "    Error: duplicate condition (" << name << ")" << std::endl;
    return false;
  } else {
    LogDebug("TriggerMenuParser") << "Added Condition " << name << " to the ConditionMap" << std::endl;
    if (corrFlag) {
      (m_corMuonTemplate[chipNr]).push_back(muonCond);
    } else {
      LogDebug("TriggerMenuParser") << "Added Condition " << name << " to the vecMuonTemplate vector" << std::endl;
      (m_vecMuonTemplate[chipNr]).push_back(muonCond);
    }
  }

  //
  return true;
}

bool l1t::TriggerMenuParser::parseMuonCorr(const L1TUtmObject* corrMu, unsigned int chipNr) {
  //    XERCES_CPP_NAMESPACE_USE
  using namespace tmeventsetup;

  // get condition, particle name (must be muon) and type name
  std::string condition = "muon";
  std::string particle = "muon";  //l1t2string( condMu.objectType() );
  std::string type = l1t2string(corrMu->getType());
  std::string name = l1t2string(corrMu->getName());
  int nrObj = 1;
  type = "1_s";
  GtConditionType cType = l1t::Type1s;

  if (nrObj < 0) {
    edm::LogError("TriggerMenuParser") << "Unknown type for muon-condition (" << type << ")"
                                       << "\nCan not determine number of trigger objects. " << std::endl;
    return false;
  }

  LogDebug("TriggerMenuParser") << "\n ****************************************** "
                                << "\n      parseMuon  "
                                << "\n condition = " << condition << "\n particle  = " << particle
                                << "\n type      = " << type << "\n name      = " << name << std::endl;

  //     // get values

  // temporary storage of the parameters
  std::vector<MuonTemplate::ObjectParameter> objParameter(nrObj);

  // Do we need this?
  MuonTemplate::CorrelationParameter corrParameter;

  // need at least two values for deltaPhi
  std::vector<uint64_t> tmpValues((nrObj > 2) ? nrObj : 2);
  tmpValues.reserve(nrObj);

  // BLW TO DO: How do we deal with these in the new format
  //    std::string str_chargeCorrelation = l1t2string( condMu.requestedChargeCorr() );
  std::string str_chargeCorrelation = "ig";
  unsigned int chargeCorrelation = 0;
  if (str_chargeCorrelation == "ig")
    chargeCorrelation = 1;
  else if (str_chargeCorrelation == "ls")
    chargeCorrelation = 2;
  else if (str_chargeCorrelation == "os")
    chargeCorrelation = 4;

  //getXMLHexTextValue("1", dst);
  corrParameter.chargeCorrelation = chargeCorrelation;  //tmpValues[0];

  // BLW TO DO: These needs to the added to the object rather than the whole condition.
  int relativeBx = 0;
  bool gEq = false;

  //const esObject* object = condMu;
  gEq = (corrMu->getComparisonOperator() == esComparisonOperator::GE);

  //  BLW TO DO: This needs to be added to the Object Parameters
  relativeBx = corrMu->getBxOffset();

  //  Loop over the cuts for this object
  int upperUnconstrainedPtInd = -1;  // Added for displaced muons
  int lowerUnconstrainedPtInd = 0;   // Added for displaced muons
  int upperImpactParameterInd = -1;  // Added for displaced muons
  int lowerImpactParameterInd = 0;   // Added for displaced muons
  int impactParameterLUT = 0xF;      // Added for displaced muons, default is to ignore unless specified
  int upperThresholdInd = -1;
  int lowerThresholdInd = 0;
  int upperIndexInd = -1;
  int lowerIndexInd = 0;
  int cntPhi = 0;
  unsigned int phiWindow1Lower = -1, phiWindow1Upper = -1, phiWindow2Lower = -1, phiWindow2Upper = -1;
  int isolationLUT = 0xF;   //default is to ignore unless specified.
  int charge = -1;          //defaut is to ignore unless specified
  int qualityLUT = 0xFFFF;  //default is to ignore unless specified.

  std::vector<MuonTemplate::Window> etaWindows;
  std::vector<MuonTemplate::Window> tfMuonIndexWindows;

  const std::vector<L1TUtmCut>& cuts = corrMu->getCuts();
  for (size_t kk = 0; kk < cuts.size(); kk++) {
    const L1TUtmCut& cut = cuts.at(kk);

    switch (cut.getCutType()) {
      case esCutType::UnconstrainedPt:  // Added for displaced muons
        lowerUnconstrainedPtInd = cut.getMinimum().index;
        upperUnconstrainedPtInd = cut.getMaximum().index;
        break;

      case esCutType::ImpactParameter:  // Added for displaced muons
        lowerImpactParameterInd = cut.getMinimum().index;
        upperImpactParameterInd = cut.getMaximum().index;
        impactParameterLUT = l1tstr2int(cut.getData());
        break;

      case esCutType::Threshold:
        lowerThresholdInd = cut.getMinimum().index;
        upperThresholdInd = cut.getMaximum().index;
        break;

      case esCutType::Slice:
        lowerIndexInd = int(cut.getMinimum().value);
        upperIndexInd = int(cut.getMaximum().value);
        break;

      case esCutType::Eta: {
        if (etaWindows.size() < 5) {
          etaWindows.push_back({cut.getMinimum().index, cut.getMaximum().index});
        } else {
          edm::LogError("TriggerMenuParser")
              << "Too Many Eta Cuts for muon-condition (" << particle << ")" << std::endl;
          return false;
        }
      } break;

      case esCutType::Phi: {
        if (cntPhi == 0) {
          phiWindow1Lower = cut.getMinimum().index;
          phiWindow1Upper = cut.getMaximum().index;
        } else if (cntPhi == 1) {
          phiWindow2Lower = cut.getMinimum().index;
          phiWindow2Upper = cut.getMaximum().index;
        } else {
          edm::LogError("TriggerMenuParser")
              << "Too Many Phi Cuts for muon-condition (" << particle << ")" << std::endl;
          return false;
        }
        cntPhi++;

      } break;

      case esCutType::Charge:
        if (cut.getData() == "positive")
          charge = 0;
        else if (cut.getData() == "negative")
          charge = 1;
        else
          charge = -1;
        break;
      case esCutType::Quality:

        qualityLUT = l1tstr2int(cut.getData());

        break;
      case esCutType::Isolation: {
        isolationLUT = l1tstr2int(cut.getData());

      } break;

      case esCutType::Index: {
        tfMuonIndexWindows.push_back({cut.getMinimum().index, cut.getMaximum().index});
      } break;

      default:
        break;
    }  //end switch

  }  //end loop over cuts

  // Set the parameter cuts
  objParameter[0].unconstrainedPtHigh = upperUnconstrainedPtInd;  // Added for displacd muons
  objParameter[0].unconstrainedPtLow = lowerUnconstrainedPtInd;   // Added for displacd muons
  objParameter[0].impactParameterHigh = upperImpactParameterInd;  // Added for displacd muons
  objParameter[0].impactParameterLow = lowerImpactParameterInd;   // Added for displacd muons
  objParameter[0].impactParameterLUT = impactParameterLUT;        // Added for displacd muons

  objParameter[0].ptHighThreshold = upperThresholdInd;
  objParameter[0].ptLowThreshold = lowerThresholdInd;

  objParameter[0].indexHigh = upperIndexInd;
  objParameter[0].indexLow = lowerIndexInd;

  objParameter[0].etaWindows = etaWindows;

  objParameter[0].phiWindow1Lower = phiWindow1Lower;
  objParameter[0].phiWindow1Upper = phiWindow1Upper;
  objParameter[0].phiWindow2Lower = phiWindow2Lower;
  objParameter[0].phiWindow2Upper = phiWindow2Upper;

  // BLW TO DO: Do we need these anymore?  Drop them?
  objParameter[0].enableMip = false;   //tmpMip[i];
  objParameter[0].enableIso = false;   //tmpEnableIso[i];
  objParameter[0].requestIso = false;  //tmpRequestIso[i];

  objParameter[0].charge = charge;
  objParameter[0].qualityLUT = qualityLUT;
  objParameter[0].isolationLUT = isolationLUT;

  objParameter[0].tfMuonIndexWindows = tfMuonIndexWindows;

  // object types - all muons
  std::vector<GlobalObject> objType(nrObj, gtMu);

  // now create a new CondMuonition
  MuonTemplate muonCond(name);

  muonCond.setCondType(cType);
  muonCond.setObjectType(objType);
  muonCond.setCondGEq(gEq);
  muonCond.setCondChipNr(chipNr);
  muonCond.setCondRelativeBx(relativeBx);
  muonCond.setConditionParameter(objParameter, corrParameter);

  if (edm::isDebugEnabled()) {
    std::ostringstream myCoutStream;
    muonCond.print(myCoutStream);
    LogTrace("TriggerMenuParser") << myCoutStream.str() << "\n" << std::endl;
  }

  /*
    // insert condition into the map and into muon template vector
    if ( !insertConditionIntoMap(muonCond, chipNr)) {
        edm::LogError("TriggerMenuParser")
                << "    Error: duplicate condition (" << name << ")"
                << std::endl;
        return false;
    }
    else {
        LogDebug("TriggerMenuParser") << "Added Condition " << name << " to the ConditionMap" << std::endl;
            (m_corMuonTemplate[chipNr]).push_back(muonCond);
    }
*/
  (m_corMuonTemplate[chipNr]).push_back(muonCond);

  //
  return true;
}

/**
 * parseMuonShower Parse a muonShower condition and insert an entry to the conditions map
 *
 * @param node The corresponding node.
 * @param name The name of the condition.
 * @param chipNr The number of the chip this condition is located.
 *
 * @return "true" if succeeded, "false" if an error occurred.
 *
 */

bool l1t::TriggerMenuParser::parseMuonShower(L1TUtmCondition condMu, unsigned int chipNr, const bool corrFlag) {
  using namespace tmeventsetup;

  // get condition, particle name (must be muon) and type name
  std::string condition = "muonShower";
  std::string particle = "muonShower";  //l1t2string( condMu.objectType() );
  std::string type = l1t2string(condMu.getType());
  std::string name = l1t2string(condMu.getName());
  // the number of muon shower objects is always 1
  int nrObj = 1;

  // condition type is always 1 particle, thus Type1s
  GtConditionType cType = l1t::Type1s;

  // temporary storage of the parameters
  std::vector<MuonShowerTemplate::ObjectParameter> objParameter(nrObj);

  if (int(condMu.getObjects().size()) != nrObj) {
    edm::LogError("TriggerMenuParser") << " condMu objects: nrObj = " << nrObj
                                       << "condMu.getObjects().size() = " << condMu.getObjects().size() << std::endl;
    return false;
  }

  // Get the muon shower object
  L1TUtmObject object = condMu.getObjects().at(0);
  int relativeBx = object.getBxOffset();

  if (condMu.getType() == esConditionType::MuonShower0) {
    objParameter[0].MuonShower0 = true;
  } else if (condMu.getType() == esConditionType::MuonShower1) {
    objParameter[0].MuonShower1 = true;
  } else if (condMu.getType() == esConditionType::MuonShower2) {
    objParameter[0].MuonShower2 = true;
  } else if (condMu.getType() == esConditionType::MuonShowerOutOfTime0) {
    objParameter[0].MuonShowerOutOfTime0 = true;
  } else if (condMu.getType() == esConditionType::MuonShowerOutOfTime1) {
    objParameter[0].MuonShowerOutOfTime1 = true;
  }

  // object types - all muons
  std::vector<GlobalObject> objType(nrObj, gtMuShower);

  // now create a new CondMuonition
  MuonShowerTemplate muonShowerCond(name);
  muonShowerCond.setCondType(cType);
  muonShowerCond.setObjectType(objType);
  muonShowerCond.setCondChipNr(chipNr);
  muonShowerCond.setCondRelativeBx(relativeBx);

  muonShowerCond.setConditionParameter(objParameter);

  if (edm::isDebugEnabled()) {
    std::ostringstream myCoutStream;
    muonShowerCond.print(myCoutStream);
  }

  // insert condition into the map and into muon template vector
  if (!insertConditionIntoMap(muonShowerCond, chipNr)) {
    edm::LogError("TriggerMenuParser") << "    Error: duplicate condition (" << name << ")" << std::endl;
    return false;
  } else {
    (m_vecMuonShowerTemplate[chipNr]).push_back(muonShowerCond);
  }

  return true;
}

/**
 * parseCalo Parse a calo condition and insert an entry to the conditions map
 *
 * @param node The corresponding node.
 * @param name The name of the condition.
 * @param chipNr The number of the chip this condition is located.
 *
 * @return "true" if succeeded, "false" if an error occurred.
 *
 */

bool l1t::TriggerMenuParser::parseCalo(L1TUtmCondition condCalo, unsigned int chipNr, const bool corrFlag) {
  //    XERCES_CPP_NAMESPACE_USE
  using namespace tmeventsetup;

  // get condition, particle name and type name

  std::string condition = "calo";
  std::string particle = "test-fix";
  std::string type = l1t2string(condCalo.getType());
  std::string name = l1t2string(condCalo.getName());

  LogDebug("TriggerMenuParser") << "\n ****************************************** "
                                << "\n      (in parseCalo) "
                                << "\n condition = " << condition << "\n particle  = " << particle
                                << "\n type      = " << type << "\n name      = " << name << std::endl;

  GtConditionType cType = l1t::TypeNull;

  // determine object type type
  // BLW TO DO:  Can this object type wait and be done later in the parsing. Or done differently completely..
  GlobalObject caloObjType;
  int nrObj = -1;

  if (condCalo.getType() == esConditionType::SingleEgamma) {
    caloObjType = gtEG;
    type = "1_s";
    cType = l1t::Type1s;
    nrObj = 1;
  } else if (condCalo.getType() == esConditionType::DoubleEgamma) {
    caloObjType = gtEG;
    type = "2_s";
    cType = l1t::Type2s;
    nrObj = 2;
  } else if (condCalo.getType() == esConditionType::TripleEgamma) {
    caloObjType = gtEG;
    cType = l1t::Type3s;
    type = "3";
    nrObj = 3;
  } else if (condCalo.getType() == esConditionType::QuadEgamma) {
    caloObjType = gtEG;
    cType = l1t::Type4s;
    type = "4";
    nrObj = 4;
  } else if (condCalo.getType() == esConditionType::SingleJet) {
    caloObjType = gtJet;
    cType = l1t::Type1s;
    type = "1_s";
    nrObj = 1;
  } else if (condCalo.getType() == esConditionType::DoubleJet) {
    caloObjType = gtJet;
    cType = l1t::Type2s;
    type = "2_s";
    nrObj = 2;
  } else if (condCalo.getType() == esConditionType::TripleJet) {
    caloObjType = gtJet;
    cType = l1t::Type3s;
    type = "3";
    nrObj = 3;
  } else if (condCalo.getType() == esConditionType::QuadJet) {
    caloObjType = gtJet;
    cType = l1t::Type4s;
    type = "4";
    nrObj = 4;
  } else if (condCalo.getType() == esConditionType::SingleTau) {
    caloObjType = gtTau;
    cType = l1t::Type1s;
    type = "1_s";
    nrObj = 1;
  } else if (condCalo.getType() == esConditionType::DoubleTau) {
    caloObjType = gtTau;
    cType = l1t::Type2s;
    type = "2_s";
    nrObj = 2;
  } else if (condCalo.getType() == esConditionType::TripleTau) {
    caloObjType = gtTau;
    cType = l1t::Type3s;
    type = "3";
    nrObj = 3;
  } else if (condCalo.getType() == esConditionType::QuadTau) {
    caloObjType = gtTau;
    cType = l1t::Type4s;
    type = "4";
    nrObj = 4;
  } else {
    edm::LogError("TriggerMenuParser") << "Wrong particle for calo-condition (" << particle << ")" << std::endl;
    return false;
  }

  //    std::string str_etComparison = l1t2string( condCalo.comparison_operator() );

  if (nrObj < 0) {
    edm::LogError("TriggerMenuParser") << "Unknown type for calo-condition (" << type << ")"
                                       << "\nCan not determine number of trigger objects. " << std::endl;
    return false;
  }

  // get values

  // temporary storage of the parameters
  std::vector<CaloTemplate::ObjectParameter> objParameter(nrObj);

  //BLW TO DO:  Can this be dropped?
  CaloTemplate::CorrelationParameter corrParameter;

  // need at least one value for deltaPhiRange
  std::vector<uint64_t> tmpValues((nrObj > 1) ? nrObj : 1);
  tmpValues.reserve(nrObj);

  if (int(condCalo.getObjects().size()) != nrObj) {
    edm::LogError("TriggerMenuParser") << " condCalo objects: nrObj = " << nrObj
                                       << "condCalo.getObjects().size() = " << condCalo.getObjects().size()
                                       << std::endl;
    return false;
  }

  //    std::string str_condCalo = "";
  //    uint64_t tempUIntH, tempUIntL;
  //    uint64_t dst;
  int cnt = 0;

  // BLW TO DO: These needs to the added to the object rather than the whole condition.
  int relativeBx = 0;
  bool gEq = false;

  // Loop over objects and extract the cuts on the objects
  const std::vector<L1TUtmObject>& objects = condCalo.getObjects();
  for (size_t jj = 0; jj < objects.size(); jj++) {
    const L1TUtmObject& object = objects.at(jj);
    gEq = (object.getComparisonOperator() == esComparisonOperator::GE);

    //  BLW TO DO: This needs to be added to the Object Parameters
    relativeBx = object.getBxOffset();

    //  Loop over the cuts for this object
    int upperThresholdInd = -1;
    int lowerThresholdInd = 0;
    int upperIndexInd = -1;
    int lowerIndexInd = 0;
    int cntPhi = 0;
    unsigned int phiWindow1Lower = -1, phiWindow1Upper = -1, phiWindow2Lower = -1, phiWindow2Upper = -1;
    int isolationLUT = 0xF;  //default is to ignore isolation unless specified.
    int qualityLUT = 0xF;    //default is to ignore quality unless specified.
    int displacedLUT = 0x0;  // Added for LLP Jets: single bit LUT: { 0 = noLLP default, 1 = LLP }
                             // Note: Currently assumes that the LSB from hwQual() getter in L1Candidate provides the
                             // (single bit) information for the displacedLUT

    std::vector<CaloTemplate::Window> etaWindows;

    const std::vector<L1TUtmCut>& cuts = object.getCuts();
    for (size_t kk = 0; kk < cuts.size(); kk++) {
      const L1TUtmCut& cut = cuts.at(kk);

      switch (cut.getCutType()) {
        case esCutType::Threshold:
          lowerThresholdInd = cut.getMinimum().index;
          upperThresholdInd = cut.getMaximum().index;
          break;
        case esCutType::Slice:
          lowerIndexInd = int(cut.getMinimum().value);
          upperIndexInd = int(cut.getMaximum().value);
          break;
        case esCutType::Eta: {
          if (etaWindows.size() < 5) {
            etaWindows.push_back({cut.getMinimum().index, cut.getMaximum().index});
          } else {
            edm::LogError("TriggerMenuParser")
                << "Too Many Eta Cuts for calo-condition (" << particle << ")" << std::endl;
            return false;
          }
        } break;

        case esCutType::Phi: {
          if (cntPhi == 0) {
            phiWindow1Lower = cut.getMinimum().index;
            phiWindow1Upper = cut.getMaximum().index;
          } else if (cntPhi == 1) {
            phiWindow2Lower = cut.getMinimum().index;
            phiWindow2Upper = cut.getMaximum().index;
          } else {
            edm::LogError("TriggerMenuParser")
                << "Too Many Phi Cuts for calo-condition (" << particle << ")" << std::endl;
            return false;
          }
          cntPhi++;

        } break;

        case esCutType::Charge: {
          edm::LogError("TriggerMenuParser") << "No charge cut for calo-condition (" << particle << ")" << std::endl;
          return false;

        } break;
        case esCutType::Quality: {
          qualityLUT = l1tstr2int(cut.getData());

        } break;
        case esCutType::Displaced: {  // Added for LLP Jets
          displacedLUT = l1tstr2int(cut.getData());

        } break;
        case esCutType::Isolation: {
          isolationLUT = l1tstr2int(cut.getData());

        } break;
        default:
          break;
      }  //end switch

    }  //end loop over cuts

    // Fill the object parameters
    objParameter[cnt].etHighThreshold = upperThresholdInd;
    objParameter[cnt].etLowThreshold = lowerThresholdInd;
    objParameter[cnt].indexHigh = upperIndexInd;
    objParameter[cnt].indexLow = lowerIndexInd;
    objParameter[cnt].etaWindows = etaWindows;
    objParameter[cnt].phiWindow1Lower = phiWindow1Lower;
    objParameter[cnt].phiWindow1Upper = phiWindow1Upper;
    objParameter[cnt].phiWindow2Lower = phiWindow2Lower;
    objParameter[cnt].phiWindow2Upper = phiWindow2Upper;
    objParameter[cnt].isolationLUT = isolationLUT;
    objParameter[cnt].qualityLUT = qualityLUT;      //TO DO: Must add
    objParameter[cnt].displacedLUT = displacedLUT;  // Added for LLP Jets

    // Output for debugging
    {
      std::ostringstream oss;
      oss << "\n      Calo ET high thresholds (hex) for calo object " << caloObjType << " " << cnt << " = " << std::hex
          << objParameter[cnt].etLowThreshold << " - " << objParameter[cnt].etHighThreshold;
      for (const auto& window : objParameter[cnt].etaWindows) {
        oss << "\n      etaWindow Lower / Upper for calo object " << cnt << " = 0x" << window.lower << " / 0x"
            << window.upper;
      }
      oss << "\n      phiWindow Lower / Upper for calo object " << cnt << " = 0x" << objParameter[cnt].phiWindow1Lower
          << " / 0x" << objParameter[cnt].phiWindow1Upper << "\n      phiWindowVeto Lower / Upper for calo object "
          << cnt << " = 0x" << objParameter[cnt].phiWindow2Lower << " / 0x" << objParameter[cnt].phiWindow2Upper
          << "\n      Isolation LUT for calo object " << cnt << " = 0x" << objParameter[cnt].isolationLUT
          << "\n      Quality LUT for calo object " << cnt << " = 0x" << objParameter[cnt].qualityLUT
          << "\n      LLP DISP LUT for calo object " << cnt << " = 0x" << objParameter[cnt].displacedLUT;
      LogDebug("TriggerMenuParser") << oss.str() << std::endl;
    }

    cnt++;
  }  //end loop over objects

  // object types - all same caloObjType
  std::vector<GlobalObject> objType(nrObj, caloObjType);

  // now create a new calo condition
  CaloTemplate caloCond(name);

  caloCond.setCondType(cType);
  caloCond.setObjectType(objType);

  //BLW TO DO: This needs to be added to the object rather than the whole condition
  caloCond.setCondGEq(gEq);
  caloCond.setCondChipNr(chipNr);

  //BLW TO DO: This needs to be added to the object rather than the whole condition
  caloCond.setCondRelativeBx(relativeBx);

  caloCond.setConditionParameter(objParameter, corrParameter);

  if (edm::isDebugEnabled()) {
    std::ostringstream myCoutStream;
    caloCond.print(myCoutStream);
    LogTrace("TriggerMenuParser") << myCoutStream.str() << "\n" << std::endl;
  }

  // insert condition into the map
  if (!insertConditionIntoMap(caloCond, chipNr)) {
    edm::LogError("TriggerMenuParser") << "    Error: duplicate condition (" << name << ")" << std::endl;

    return false;
  } else {
    if (corrFlag) {
      (m_corCaloTemplate[chipNr]).push_back(caloCond);
    } else {
      (m_vecCaloTemplate[chipNr]).push_back(caloCond);
    }
  }

  //
  return true;
}

/**
 * parseCalo Parse a calo condition and insert an entry to the conditions map
 *
 * @param node The corresponding node.
 * @param name The name of the condition.
 * @param chipNr The number of the chip this condition is located.
 *
 * @return "true" if succeeded, "false" if an error occurred.
 *
 */

bool l1t::TriggerMenuParser::parseCaloCorr(const L1TUtmObject* corrCalo, unsigned int chipNr) {
  //    XERCES_CPP_NAMESPACE_USE
  using namespace tmeventsetup;

  // get condition, particle name and type name

  std::string condition = "calo";
  std::string particle = "test-fix";
  std::string type = l1t2string(corrCalo->getType());
  std::string name = l1t2string(corrCalo->getName());

  LogDebug("TriggerMenuParser") << "\n ****************************************** "
                                << "\n      (in parseCalo) "
                                << "\n condition = " << condition << "\n particle  = " << particle
                                << "\n type      = " << type << "\n name      = " << name << std::endl;

  // determine object type type
  // BLW TO DO:  Can this object type wait and be done later in the parsing. Or done differently completely..
  GlobalObject caloObjType;
  int nrObj = 1;
  type = "1_s";
  GtConditionType cType = l1t::Type1s;

  if (corrCalo->getType() == esObjectType::Egamma) {
    caloObjType = gtEG;
  } else if (corrCalo->getType() == esObjectType::Jet) {
    caloObjType = gtJet;
  } else if (corrCalo->getType() == esObjectType::Tau) {
    caloObjType = gtTau;
  } else {
    edm::LogError("TriggerMenuParser") << "Wrong particle for calo-condition (" << particle << ")" << std::endl;
    return false;
  }

  //    std::string str_etComparison = l1t2string( condCalo.comparison_operator() );

  if (nrObj < 0) {
    edm::LogError("TriggerMenuParser") << "Unknown type for calo-condition (" << type << ")"
                                       << "\nCan not determine number of trigger objects. " << std::endl;
    return false;
  }

  // get values

  // temporary storage of the parameters
  std::vector<CaloTemplate::ObjectParameter> objParameter(nrObj);

  //BLW TO DO:  Can this be dropped?
  CaloTemplate::CorrelationParameter corrParameter;

  // need at least one value for deltaPhiRange
  std::vector<uint64_t> tmpValues((nrObj > 1) ? nrObj : 1);
  tmpValues.reserve(nrObj);

  // BLW TO DO: These needs to the added to the object rather than the whole condition.
  int relativeBx = 0;
  bool gEq = false;

  gEq = (corrCalo->getComparisonOperator() == esComparisonOperator::GE);

  //  BLW TO DO: This needs to be added to the Object Parameters
  relativeBx = corrCalo->getBxOffset();

  //  Loop over the cuts for this object
  int upperThresholdInd = -1;
  int lowerThresholdInd = 0;
  int upperIndexInd = -1;
  int lowerIndexInd = 0;
  int cntPhi = 0;
  unsigned int phiWindow1Lower = -1, phiWindow1Upper = -1, phiWindow2Lower = -1, phiWindow2Upper = -1;
  int isolationLUT = 0xF;  //default is to ignore isolation unless specified.
  int qualityLUT = 0xF;    //default is to ignore quality unless specified.
  int displacedLUT = 0x0;  // Added for LLP Jets:  single bit LUT:  { 0 = noLLP default, 1 = LLP }
                           // Note:  Currently assume that the hwQual() getter in L1Candidate provides the
                           //        (single bit) information for the displacedLUT

  std::vector<CaloTemplate::Window> etaWindows;

  const std::vector<L1TUtmCut>& cuts = corrCalo->getCuts();
  for (size_t kk = 0; kk < cuts.size(); kk++) {
    const L1TUtmCut& cut = cuts.at(kk);

    switch (cut.getCutType()) {
      case esCutType::Threshold:
        lowerThresholdInd = cut.getMinimum().index;
        upperThresholdInd = cut.getMaximum().index;
        break;
      case esCutType::Slice:
        lowerIndexInd = int(cut.getMinimum().value);
        upperIndexInd = int(cut.getMaximum().value);
        break;
      case esCutType::Eta: {
        if (etaWindows.size() < 5) {
          etaWindows.push_back({cut.getMinimum().index, cut.getMaximum().index});
        } else {
          edm::LogError("TriggerMenuParser")
              << "Too Many Eta Cuts for calo-condition (" << particle << ")" << std::endl;
          return false;
        }
      } break;

      case esCutType::Phi: {
        if (cntPhi == 0) {
          phiWindow1Lower = cut.getMinimum().index;
          phiWindow1Upper = cut.getMaximum().index;
        } else if (cntPhi == 1) {
          phiWindow2Lower = cut.getMinimum().index;
          phiWindow2Upper = cut.getMaximum().index;
        } else {
          edm::LogError("TriggerMenuParser")
              << "Too Many Phi Cuts for calo-condition (" << particle << ")" << std::endl;
          return false;
        }
        cntPhi++;

      } break;

      case esCutType::Charge: {
        edm::LogError("TriggerMenuParser") << "No charge cut for calo-condition (" << particle << ")" << std::endl;
        return false;

      } break;
      case esCutType::Quality: {
        qualityLUT = l1tstr2int(cut.getData());

      } break;
      case esCutType::Displaced: {  // Added for LLP Jets
        displacedLUT = l1tstr2int(cut.getData());

      } break;
      case esCutType::Isolation: {
        isolationLUT = l1tstr2int(cut.getData());

      } break;
      default:
        break;
    }  //end switch

  }  //end loop over cuts

  // Fill the object parameters
  objParameter[0].etLowThreshold = lowerThresholdInd;
  objParameter[0].etHighThreshold = upperThresholdInd;
  objParameter[0].indexHigh = upperIndexInd;
  objParameter[0].indexLow = lowerIndexInd;
  objParameter[0].etaWindows = etaWindows;
  objParameter[0].phiWindow1Lower = phiWindow1Lower;
  objParameter[0].phiWindow1Upper = phiWindow1Upper;
  objParameter[0].phiWindow2Lower = phiWindow2Lower;
  objParameter[0].phiWindow2Upper = phiWindow2Upper;
  objParameter[0].isolationLUT = isolationLUT;
  objParameter[0].qualityLUT = qualityLUT;      //TO DO: Must add
  objParameter[0].displacedLUT = displacedLUT;  // Added for LLP Jets

  // Output for debugging
  {
    std::ostringstream oss;
    oss << "\n      Calo ET high threshold (hex) for calo object " << caloObjType << " "
        << " = " << std::hex << objParameter[0].etLowThreshold << " - " << objParameter[0].etHighThreshold;
    for (const auto& window : objParameter[0].etaWindows) {
      oss << "\n      etaWindow Lower / Upper for calo object "
          << " = 0x" << window.lower << " / 0x" << window.upper;
    }
    oss << "\n      phiWindow Lower / Upper for calo object "
        << " = 0x" << objParameter[0].phiWindow1Lower << " / 0x" << objParameter[0].phiWindow1Upper
        << "\n      phiWindowVeto Lower / Upper for calo object "
        << " = 0x" << objParameter[0].phiWindow2Lower << " / 0x" << objParameter[0].phiWindow2Upper
        << "\n      Isolation LUT for calo object "
        << " = 0x" << objParameter[0].isolationLUT << "\n      Quality LUT for calo object "
        << " = 0x" << objParameter[0].qualityLUT << "\n      LLP DISP LUT for calo object "
        << " = 0x" << objParameter[0].displacedLUT;
    LogDebug("TriggerMenuParser") << oss.str() << std::endl;
  }

  // object types - all same caloObjType
  std::vector<GlobalObject> objType(nrObj, caloObjType);

  // now create a new calo condition
  CaloTemplate caloCond(name);

  caloCond.setCondType(cType);
  caloCond.setObjectType(objType);

  //BLW TO DO: This needs to be added to the object rather than the whole condition
  caloCond.setCondGEq(gEq);
  caloCond.setCondChipNr(chipNr);

  //BLW TO DO: This needs to be added to the object rather than the whole condition
  caloCond.setCondRelativeBx(relativeBx);

  caloCond.setConditionParameter(objParameter, corrParameter);

  if (edm::isDebugEnabled()) {
    std::ostringstream myCoutStream;
    caloCond.print(myCoutStream);
    LogTrace("TriggerMenuParser") << myCoutStream.str() << "\n" << std::endl;
  }

  /*
    // insert condition into the map
    if ( !insertConditionIntoMap(caloCond, chipNr)) {

        edm::LogError("TriggerMenuParser")
                << "    Error: duplicate condition (" << name << ")"
                << std::endl;

        return false;
    }
    else {
            (m_corCaloTemplate[chipNr]).push_back(caloCond);
    }
*/
  (m_corCaloTemplate[chipNr]).push_back(caloCond);

  //
  return true;
}

/**
 * parseEnergySum Parse an "energy sum" condition and insert an entry to the conditions map
 *
 * @param node The corresponding node.
 * @param name The name of the condition.
 * @param chipNr The number of the chip this condition is located.
 *
 * @return "true" if succeeded, "false" if an error occurred.
 *
 */

bool l1t::TriggerMenuParser::parseEnergySum(L1TUtmCondition condEnergySum, unsigned int chipNr, const bool corrFlag) {
  //    XERCES_CPP_NAMESPACE_USE
  using namespace tmeventsetup;

  // get condition, particle name and type name

  std::string condition = "calo";
  std::string type = l1t2string(condEnergySum.getType());
  std::string name = l1t2string(condEnergySum.getName());

  LogDebug("TriggerMenuParser") << "\n ****************************************** "
                                << "\n      (in parseEnergySum) "
                                << "\n condition = " << condition << "\n type      = " << type
                                << "\n name      = " << name << std::endl;

  // determine object type type
  GlobalObject energySumObjType;
  GtConditionType cType;

  if (condEnergySum.getType() == esConditionType::MissingEt) {
    energySumObjType = GlobalObject::gtETM;
    cType = TypeETM;
  } else if (condEnergySum.getType() == esConditionType::TotalEt) {
    energySumObjType = GlobalObject::gtETT;
    cType = TypeETT;
  } else if (condEnergySum.getType() == esConditionType::TotalEtEM) {
    energySumObjType = GlobalObject::gtETTem;
    cType = TypeETTem;
  } else if (condEnergySum.getType() == esConditionType::TotalHt) {
    energySumObjType = GlobalObject::gtHTT;
    cType = TypeHTT;
  } else if (condEnergySum.getType() == esConditionType::MissingHt) {
    energySumObjType = GlobalObject::gtHTM;
    cType = TypeHTM;
  } else if (condEnergySum.getType() == esConditionType::MissingEtHF) {
    energySumObjType = GlobalObject::gtETMHF;
    cType = TypeETMHF;
  } else if (condEnergySum.getType() == esConditionType::TowerCount) {
    energySumObjType = GlobalObject::gtTowerCount;
    cType = TypeTowerCount;
  } else if (condEnergySum.getType() == esConditionType::MinBiasHFP0) {
    energySumObjType = GlobalObject::gtMinBiasHFP0;
    cType = TypeMinBiasHFP0;
  } else if (condEnergySum.getType() == esConditionType::MinBiasHFM0) {
    energySumObjType = GlobalObject::gtMinBiasHFM0;
    cType = TypeMinBiasHFM0;
  } else if (condEnergySum.getType() == esConditionType::MinBiasHFP1) {
    energySumObjType = GlobalObject::gtMinBiasHFP1;
    cType = TypeMinBiasHFP1;
  } else if (condEnergySum.getType() == esConditionType::MinBiasHFM1) {
    energySumObjType = GlobalObject::gtMinBiasHFM1;
    cType = TypeMinBiasHFM1;
  } else if (condEnergySum.getType() == esConditionType::AsymmetryEt) {
    energySumObjType = GlobalObject::gtAsymmetryEt;
    cType = TypeAsymEt;
  } else if (condEnergySum.getType() == esConditionType::AsymmetryHt) {
    energySumObjType = GlobalObject::gtAsymmetryHt;
    cType = TypeAsymHt;
  } else if (condEnergySum.getType() == esConditionType::AsymmetryEtHF) {
    energySumObjType = GlobalObject::gtAsymmetryEtHF;
    cType = TypeAsymEtHF;
  } else if (condEnergySum.getType() == esConditionType::AsymmetryHtHF) {
    energySumObjType = GlobalObject::gtAsymmetryHtHF;
    cType = TypeAsymHtHF;
  } else if (condEnergySum.getType() == esConditionType::Centrality0) {
    energySumObjType = GlobalObject::gtCentrality0;
    cType = TypeCent0;
  } else if (condEnergySum.getType() == esConditionType::Centrality1) {
    energySumObjType = GlobalObject::gtCentrality1;
    cType = TypeCent1;
  } else if (condEnergySum.getType() == esConditionType::Centrality2) {
    energySumObjType = GlobalObject::gtCentrality2;
    cType = TypeCent2;
  } else if (condEnergySum.getType() == esConditionType::Centrality3) {
    energySumObjType = GlobalObject::gtCentrality3;
    cType = TypeCent3;
  } else if (condEnergySum.getType() == esConditionType::Centrality4) {
    energySumObjType = GlobalObject::gtCentrality4;
    cType = TypeCent4;
  } else if (condEnergySum.getType() == esConditionType::Centrality5) {
    energySumObjType = GlobalObject::gtCentrality5;
    cType = TypeCent5;
  } else if (condEnergySum.getType() == esConditionType::Centrality6) {
    energySumObjType = GlobalObject::gtCentrality6;
    cType = TypeCent6;
  } else if (condEnergySum.getType() == esConditionType::Centrality7) {
    energySumObjType = GlobalObject::gtCentrality7;
    cType = TypeCent7;
  } else {
    edm::LogError("TriggerMenuParser") << "Wrong type for energy-sum condition (" << type << ")" << std::endl;
    return false;
  }

  // global object
  int nrObj = 1;

  //    std::string str_etComparison = l1t2string( condEnergySum.comparison_operator() );

  // get values

  // temporary storage of the parameters
  std::vector<EnergySumTemplate::ObjectParameter> objParameter(nrObj);

  int cnt = 0;

  // BLW TO DO: These needs to the added to the object rather than the whole condition.
  int relativeBx = 0;
  bool gEq = false;

  //    l1t::EnergySumsObjectRequirement objPar = condEnergySum.objectRequirement();

  // Loop over objects and extract the cuts on the objects
  const std::vector<L1TUtmObject>& objects = condEnergySum.getObjects();
  for (size_t jj = 0; jj < objects.size(); jj++) {
    const L1TUtmObject& object = objects.at(jj);
    gEq = (object.getComparisonOperator() == esComparisonOperator::GE);

    //  BLW TO DO: This needs to be added to the Object Parameters
    relativeBx = object.getBxOffset();

    //  Loop over the cuts for this object
    int lowerThresholdInd = 0;
    int upperThresholdInd = -1;
    int cntPhi = 0;
    unsigned int phiWindow1Lower = -1, phiWindow1Upper = -1, phiWindow2Lower = -1, phiWindow2Upper = -1;

    const std::vector<L1TUtmCut>& cuts = object.getCuts();
    for (size_t kk = 0; kk < cuts.size(); kk++) {
      const L1TUtmCut& cut = cuts.at(kk);

      switch (cut.getCutType()) {
        case esCutType::Threshold:
          lowerThresholdInd = cut.getMinimum().index;
          upperThresholdInd = cut.getMaximum().index;
          break;

        case esCutType::Eta:
          break;

        case esCutType::Phi: {
          if (cntPhi == 0) {
            phiWindow1Lower = cut.getMinimum().index;
            phiWindow1Upper = cut.getMaximum().index;
          } else if (cntPhi == 1) {
            phiWindow2Lower = cut.getMinimum().index;
            phiWindow2Upper = cut.getMaximum().index;
          } else {
            edm::LogError("TriggerMenuParser") << "Too Many Phi Cuts for esum-condition (" << type << ")" << std::endl;
            return false;
          }
          cntPhi++;

        } break;

        case esCutType::Count:
          lowerThresholdInd = cut.getMinimum().index;
          upperThresholdInd = 0xffffff;
          break;

        default:
          break;
      }  //end switch

    }  //end loop over cuts

    // Fill the object parameters
    objParameter[cnt].etLowThreshold = lowerThresholdInd;
    objParameter[cnt].etHighThreshold = upperThresholdInd;
    objParameter[cnt].phiWindow1Lower = phiWindow1Lower;
    objParameter[cnt].phiWindow1Upper = phiWindow1Upper;
    objParameter[cnt].phiWindow2Lower = phiWindow2Lower;
    objParameter[cnt].phiWindow2Upper = phiWindow2Upper;

    // Output for debugging
    LogDebug("TriggerMenuParser") << "\n      EnergySum ET high threshold (hex) for energy sum object " << cnt << " = "
                                  << std::hex << objParameter[cnt].etLowThreshold << " - "
                                  << objParameter[cnt].etHighThreshold
                                  << "\n      phiWindow Lower / Upper for calo object " << cnt << " = 0x"
                                  << objParameter[cnt].phiWindow1Lower << " / 0x" << objParameter[cnt].phiWindow1Upper
                                  << "\n      phiWindowVeto Lower / Upper for calo object " << cnt << " = 0x"
                                  << objParameter[cnt].phiWindow2Lower << " / 0x" << objParameter[cnt].phiWindow2Upper
                                  << std::dec << std::endl;

    cnt++;
  }  //end loop over objects

  // object types - all same energySumObjType
  std::vector<GlobalObject> objType(nrObj, energySumObjType);

  // now create a new energySum condition

  EnergySumTemplate energySumCond(name);

  energySumCond.setCondType(cType);
  energySumCond.setObjectType(objType);
  energySumCond.setCondGEq(gEq);
  energySumCond.setCondChipNr(chipNr);
  energySumCond.setCondRelativeBx(relativeBx);

  energySumCond.setConditionParameter(objParameter);

  if (edm::isDebugEnabled()) {
    std::ostringstream myCoutStream;
    energySumCond.print(myCoutStream);
    LogTrace("TriggerMenuParser") << myCoutStream.str() << "\n" << std::endl;
  }

  // insert condition into the map
  if (!insertConditionIntoMap(energySumCond, chipNr)) {
    edm::LogError("TriggerMenuParser") << "    Error: duplicate condition (" << name << ")" << std::endl;

    return false;
  } else {
    if (corrFlag) {
      (m_corEnergySumTemplate[chipNr]).push_back(energySumCond);

    } else {
      (m_vecEnergySumTemplate[chipNr]).push_back(energySumCond);
    }
  }

  //
  return true;
}

/**
 * parseEnergySumZdc Parse an "energy sum" condition from the ZDC subsystem and insert an entry to the conditions map
 *
 * @param node The corresponding node.
 * @param name The name of the condition.
 * @param chipNr The number of the chip this condition is located.
 *
 * @return "true" if succeeded, "false" if an error occurred.
 *
 */

bool l1t::TriggerMenuParser::parseEnergySumZdc(L1TUtmCondition condEnergySumZdc,
                                               unsigned int chipNr,
                                               const bool corrFlag) {
  //    XERCES_CPP_NAMESPACE_USE
  using namespace tmeventsetup;

  // get condition, particle name and type name

  std::string condition = "calo";
  std::string type = l1t2string(condEnergySumZdc.getType());
  std::string name = l1t2string(condEnergySumZdc.getName());

  LogDebug("TriggerMenuParser")
      << "\n ******************************************\n      (in parseEnergySumZdc)\n condition = " << condition
      << "\n type      = " << type << "\n name      = " << name;

  // determine object type
  GlobalObject energySumObjType;
  GtConditionType cType;

  if (condEnergySumZdc.getType() == esConditionType::ZDCPlus) {
    LogDebug("TriggerMenuParser") << "ZDC signals: esConditionType::ZDCPlus " << std::endl;
    energySumObjType = GlobalObject::gtZDCP;
    cType = TypeZDCP;
  } else if (condEnergySumZdc.getType() == esConditionType::ZDCMinus) {
    LogDebug("TriggerMenuParser") << "ZDC signals: esConditionType::ZDCMinus " << std::endl;
    energySumObjType = GlobalObject::gtZDCM;
    cType = TypeZDCM;
  } else {
    edm::LogError("TriggerMenuParser") << "Wrong type for ZDC energy-sum condition (" << type << ")" << std::endl;
    return false;
  }

  // global object
  int nrObj = 1;

  // temporary storage of the parameters
  std::vector<EnergySumZdcTemplate::ObjectParameter> objParameter(nrObj);

  //  Loop over the cuts for this object
  int lowerThresholdInd = 0;
  int upperThresholdInd = -1;

  int cnt = 0;

  // BLW TO DO: These needs to the added to the object rather than the whole condition.
  int relativeBx = 0;
  bool gEq = false;

  //    l1t::EnergySumsObjectRequirement objPar = condEnergySumZdc.objectRequirement();

  // Loop over objects and extract the cuts on the objects
  const std::vector<L1TUtmObject>& objects = condEnergySumZdc.getObjects();
  for (size_t jj = 0; jj < objects.size(); jj++) {
    const L1TUtmObject& object = objects.at(jj);
    gEq = (object.getComparisonOperator() == esComparisonOperator::GE);

    //  BLW TO DO: This needs to be added to the Object Parameters
    relativeBx = object.getBxOffset();

    //  Loop over the cuts for this object
    const std::vector<L1TUtmCut>& cuts = object.getCuts();
    for (size_t kk = 0; kk < cuts.size(); kk++) {
      const L1TUtmCut& cut = cuts.at(kk);

      switch (cut.getCutType()) {
        case esCutType::Threshold:
          lowerThresholdInd = cut.getMinimum().index;
          upperThresholdInd = cut.getMaximum().index;
          break;

        case esCutType::Count:
          lowerThresholdInd = cut.getMinimum().index;
          upperThresholdInd = 0xffffff;
          break;

        default:
          break;
      }  //end switch

    }  //end loop over cuts

    // Fill the object parameters
    objParameter[cnt].etLowThreshold = lowerThresholdInd;
    objParameter[cnt].etHighThreshold = upperThresholdInd;

    // Output for debugging
    LogDebug("TriggerMenuParser") << "\n      EnergySumZdc ET high threshold (hex) for energy sum object " << cnt
                                  << " = " << std::hex << objParameter[cnt].etLowThreshold << " - "
                                  << objParameter[cnt].etHighThreshold << std::dec;

    cnt++;
  }  //end loop over objects

  // object types - all same energySumObjType
  std::vector<GlobalObject> objType(nrObj, energySumObjType);

  // now create a new energySum condition

  EnergySumZdcTemplate energySumCond(name);

  energySumCond.setCondType(cType);
  energySumCond.setObjectType(objType);
  energySumCond.setCondGEq(gEq);
  energySumCond.setCondChipNr(chipNr);
  energySumCond.setCondRelativeBx(relativeBx);

  energySumCond.setConditionParameter(objParameter);

  if (edm::isDebugEnabled()) {
    std::ostringstream myCoutStream;
    energySumCond.print(myCoutStream);
    LogTrace("TriggerMenuParser") << myCoutStream.str() << "\n" << std::endl;
  }

  // insert condition into the map
  if (!insertConditionIntoMap(energySumCond, chipNr)) {
    edm::LogError("TriggerMenuParser") << "    Error: duplicate condition (" << name << ")" << std::endl;

    return false;
  } else {
    (m_vecEnergySumZdcTemplate[chipNr]).push_back(energySumCond);
  }
  //
  return true;
}

/**
 * parseEnergySumCorr Parse an "energy sum" correlation condition and insert an entry to the conditions map
 *
 * @param node The corresponding node.
 * @param name The name of the condition.
 * @param chipNr The number of the chip this condition is located.
 *
 * @return "true" if succeeded, "false" if an error occurred.
 *
 */

bool l1t::TriggerMenuParser::parseEnergySumCorr(const L1TUtmObject* corrESum, unsigned int chipNr) {
  //    XERCES_CPP_NAMESPACE_USE
  using namespace tmeventsetup;

  // get condition, particle name and type name

  std::string condition = "calo";
  std::string type = l1t2string(corrESum->getType());
  std::string name = l1t2string(corrESum->getName());

  LogDebug("TriggerMenuParser") << "\n ****************************************** "
                                << "\n      (in parseEnergySum) "
                                << "\n condition = " << condition << "\n type      = " << type
                                << "\n name      = " << name << std::endl;

  // determine object type type
  GlobalObject energySumObjType;
  GtConditionType cType;

  if (corrESum->getType() == esObjectType::ETM) {
    energySumObjType = GlobalObject::gtETM;
    cType = TypeETM;
  } else if (corrESum->getType() == esObjectType::HTM) {
    energySumObjType = GlobalObject::gtHTM;
    cType = TypeHTM;
  } else if (corrESum->getType() == esObjectType::ETMHF) {
    energySumObjType = GlobalObject::gtETMHF;
    cType = TypeETMHF;
  } else if (corrESum->getType() == esObjectType::TOWERCOUNT) {
    energySumObjType = GlobalObject::gtTowerCount;
    cType = TypeTowerCount;
  } else {
    edm::LogError("TriggerMenuParser") << "Wrong type for energy-sum correclation condition (" << type << ")"
                                       << std::endl;
    return false;
  }

  // global object
  int nrObj = 1;

  //    std::string str_etComparison = l1t2string( condEnergySum.comparison_operator() );

  // get values

  // temporary storage of the parameters
  std::vector<EnergySumTemplate::ObjectParameter> objParameter(nrObj);

  int cnt = 0;

  // BLW TO DO: These needs to the added to the object rather than the whole condition.
  int relativeBx = 0;
  bool gEq = false;

  //    l1t::EnergySumsObjectRequirement objPar = condEnergySum.objectRequirement();

  gEq = (corrESum->getComparisonOperator() == esComparisonOperator::GE);

  //  BLW TO DO: This needs to be added to the Object Parameters
  relativeBx = corrESum->getBxOffset();

  //  Loop over the cuts for this object
  int lowerThresholdInd = 0;
  int upperThresholdInd = -1;
  int cntPhi = 0;
  unsigned int phiWindow1Lower = -1, phiWindow1Upper = -1, phiWindow2Lower = -1, phiWindow2Upper = -1;

  const std::vector<L1TUtmCut>& cuts = corrESum->getCuts();
  for (size_t kk = 0; kk < cuts.size(); kk++) {
    const L1TUtmCut& cut = cuts.at(kk);

    switch (cut.getCutType()) {
      case esCutType::Threshold:
        lowerThresholdInd = cut.getMinimum().index;
        upperThresholdInd = cut.getMaximum().index;
        break;

      case esCutType::Eta:
        break;

      case esCutType::Phi: {
        if (cntPhi == 0) {
          phiWindow1Lower = cut.getMinimum().index;
          phiWindow1Upper = cut.getMaximum().index;
        } else if (cntPhi == 1) {
          phiWindow2Lower = cut.getMinimum().index;
          phiWindow2Upper = cut.getMaximum().index;
        } else {
          edm::LogError("TriggerMenuParser") << "Too Many Phi Cuts for esum-condition (" << type << ")" << std::endl;
          return false;
        }
        cntPhi++;

      } break;

      default:
        break;
    }  //end switch

  }  //end loop over cuts

  // Fill the object parameters
  objParameter[0].etLowThreshold = lowerThresholdInd;
  objParameter[0].etHighThreshold = upperThresholdInd;
  objParameter[0].phiWindow1Lower = phiWindow1Lower;
  objParameter[0].phiWindow1Upper = phiWindow1Upper;
  objParameter[0].phiWindow2Lower = phiWindow2Lower;
  objParameter[0].phiWindow2Upper = phiWindow2Upper;

  // Output for debugging
  LogDebug("TriggerMenuParser") << "\n      EnergySum ET high threshold (hex) for energy sum object " << cnt << " = "
                                << std::hex << objParameter[0].etLowThreshold << " - " << objParameter[0].etLowThreshold
                                << "\n      phiWindow Lower / Upper for calo object " << cnt << " = 0x"
                                << objParameter[0].phiWindow1Lower << " / 0x" << objParameter[0].phiWindow1Upper
                                << "\n      phiWindowVeto Lower / Upper for calo object " << cnt << " = 0x"
                                << objParameter[0].phiWindow2Lower << " / 0x" << objParameter[0].phiWindow2Upper
                                << std::dec << std::endl;

  // object types - all same energySumObjType
  std::vector<GlobalObject> objType(nrObj, energySumObjType);

  // now create a new energySum condition

  EnergySumTemplate energySumCond(name);

  energySumCond.setCondType(cType);
  energySumCond.setObjectType(objType);
  energySumCond.setCondGEq(gEq);
  energySumCond.setCondChipNr(chipNr);
  energySumCond.setCondRelativeBx(relativeBx);

  energySumCond.setConditionParameter(objParameter);

  if (edm::isDebugEnabled()) {
    std::ostringstream myCoutStream;
    energySumCond.print(myCoutStream);
    LogTrace("TriggerMenuParser") << myCoutStream.str() << "\n" << std::endl;
  }
  /*
    // insert condition into the map
    if ( !insertConditionIntoMap(energySumCond, chipNr)) {

        edm::LogError("TriggerMenuParser")
                << "    Error: duplicate condition (" << name << ")"
                << std::endl;

        return false;
    }
    else {

       (m_corEnergySumTemplate[chipNr]).push_back(energySumCond);

    }
*/
  (m_corEnergySumTemplate[chipNr]).push_back(energySumCond);

  //
  return true;
}

/**                                                                                                                                                            
 * parseEnergySumCorr Parse an "energy sum" correlation condition and insert an entry to the conditions map                                                    
 *                                                                                                                                                             
 * @param node The corresponding node.                                                                                                                        
 * @param name The name of the condition.                                                                                                                      
 * @param chipNr The number of the chip this condition is located.                                                                                             
 *                                                                                                                                                             
 * @return "true" if succeeded, "false" if an error occurred.                                                                                                  
 *                                                                                                                                                             
 */

bool l1t::TriggerMenuParser::parseAXOL1TL(L1TUtmCondition condAXOL1TL, unsigned int chipNr) {
  using namespace tmeventsetup;

  // get condition, particle name and particle type
  std::string condition = "axol1tl";
  std::string type = l1t2string(condAXOL1TL.getType());
  std::string name = l1t2string(condAXOL1TL.getName());

  LogDebug("TriggerMenuParser") << " ****************************************** " << std::endl
                                << "     (in parseAXOL1TL) " << std::endl
                                << " condition = " << condition
                                << std::endl
                                // << " particle  = " << particle << std::endl
                                << " type      = " << type << std::endl
                                << " name      = " << name << std::endl;

  int nrObj = 1;
  GtConditionType cType = TypeAXOL1TL;

  std::vector<AXOL1TLTemplate::ObjectParameter> objParameter(nrObj);

  if (int(condAXOL1TL.getObjects().size()) != nrObj) {
    edm::LogError("TriggerMenuParser") << " condAXOL1TL objects: nrObj = " << nrObj
                                       << "condAXOL1TL.getObjects().size() = " << condAXOL1TL.getObjects().size()
                                       << std::endl;
    return false;
  }

  // Get the axol1tl object
  L1TUtmObject object = condAXOL1TL.getObjects().at(0);
  int relativeBx = object.getBxOffset();
  bool gEq = (object.getComparisonOperator() == esComparisonOperator::GE);

  //Loop over cuts for this  object
  int lowerThresholdInd = 0;
  int upperThresholdInd = -1;

  const std::vector<L1TUtmCut>& cuts = object.getCuts();
  for (size_t kk = 0; kk < cuts.size(); kk++) {
    const L1TUtmCut& cut = cuts.at(kk);

    switch (cut.getCutType()) {
      case esCutType::AnomalyScore:
        lowerThresholdInd = cut.getMinimum().value;
        upperThresholdInd = cut.getMaximum().value;
        break;
      default:
        break;
    }  //end switch
  }    //end cut loop

  //fill object params
  objParameter[0].minAXOL1TLThreshold = lowerThresholdInd;
  objParameter[0].maxAXOL1TLThreshold = upperThresholdInd;

  // create a new AXOL1TL  condition
  AXOL1TLTemplate axol1tlCond(name);
  axol1tlCond.setCondType(cType);
  axol1tlCond.setCondGEq(gEq);
  axol1tlCond.setCondChipNr(chipNr);
  axol1tlCond.setCondRelativeBx(relativeBx);
  axol1tlCond.setConditionParameter(objParameter);

  if (edm::isDebugEnabled()) {
    std::ostringstream myCoutStream;
    axol1tlCond.print(myCoutStream);
    LogTrace("TriggerMenuParser") << myCoutStream.str() << "\n" << std::endl;
  }

  // check that the condition does not exist already in the map
  if (!insertConditionIntoMap(axol1tlCond, chipNr)) {
    edm::LogError("TriggerMenuParser") << "    Error: duplicate AXOL1TL condition (" << name << ")" << std::endl;
    return false;
  }

  (m_vecAXOL1TLTemplate[chipNr]).push_back(axol1tlCond);

  return true;
}

/**
 * parseExternal Parse an External condition and
 * insert an entry to the conditions map
 *
 * @param node The corresponding node.
 * @param name The name of the condition.
 * @param chipNr The number of the chip this condition is located.
 *
 * @return "true" if succeeded, "false" if an error occurred.
 *
 */

bool l1t::TriggerMenuParser::parseExternal(L1TUtmCondition condExt, unsigned int chipNr) {
  using namespace tmeventsetup;
  // get condition, particle name and type name
  std::string condition = "ext";
  std::string particle = "test-fix";
  std::string type = l1t2string(condExt.getType());
  std::string name = l1t2string(condExt.getName());

  LogDebug("TriggerMenuParser") << "\n ****************************************** "
                                << "\n      (in parseExternal) "
                                << "\n condition = " << condition << "\n particle  = " << particle
                                << "\n type      = " << type << "\n name      = " << name << std::endl;

  // object type and condition type
  // object type - irrelevant for External conditions
  GtConditionType cType = TypeExternal;
  GlobalObject extSignalType = GlobalObject::gtExternal;
  int nrObj = 1;  //only one object for these conditions

  int relativeBx = 0;
  unsigned int channelID = 0;

  // Get object for External conditions
  const std::vector<L1TUtmObject>& objects = condExt.getObjects();
  for (size_t jj = 0; jj < objects.size(); jj++) {
    const L1TUtmObject& object = objects.at(jj);
    if (object.getType() == esObjectType::EXT) {
      relativeBx = object.getBxOffset();
      channelID = object.getExternalChannelId();
    }
  }

  // set the boolean value for the ge_eq mode - irrelevant for External conditions
  bool gEq = false;

  //object types - all same for external conditions
  std::vector<GlobalObject> objType(nrObj, extSignalType);

  // now create a new External condition
  ExternalTemplate externalCond(name);

  externalCond.setCondType(cType);
  externalCond.setObjectType(objType);
  externalCond.setCondGEq(gEq);
  externalCond.setCondChipNr(chipNr);
  externalCond.setCondRelativeBx(relativeBx);
  externalCond.setExternalChannel(channelID);

  LogTrace("TriggerMenuParser") << externalCond << "\n" << std::endl;

  // insert condition into the map
  if (!insertConditionIntoMap(externalCond, chipNr)) {
    edm::LogError("TriggerMenuParser") << "    Error: duplicate condition (" << name << ")" << std::endl;

    return false;
  } else {
    (m_vecExternalTemplate[chipNr]).push_back(externalCond);
  }

  return true;
}

/**
 * parseCorrelation Parse a correlation condition and
 * insert an entry to the conditions map
 *
 * @param node The corresponding node.
 * @param name The name of the condition.
 * @param chipNr The number of the chip this condition is located.
 *
 * @return "true" if succeeded, "false" if an error occurred.
 *
 */

bool l1t::TriggerMenuParser::parseCorrelation(L1TUtmCondition corrCond, unsigned int chipNr) {
  using namespace tmeventsetup;
  std::string condition = "corr";
  std::string particle = "test-fix";
  std::string type = l1t2string(corrCond.getType());
  std::string name = l1t2string(corrCond.getName());

  LogDebug("TriggerMenuParser") << " ****************************************** " << std::endl
                                << "     (in parseCorrelation) " << std::endl
                                << " condition = " << condition << std::endl
                                << " particle  = " << particle << std::endl
                                << " type      = " << type << std::endl
                                << " name      = " << name << std::endl;

  // create a new correlation condition
  CorrelationTemplate correlationCond(name);

  // check that the condition does not exist already in the map
  if (!insertConditionIntoMap(correlationCond, chipNr)) {
    edm::LogError("TriggerMenuParser") << "    Error: duplicate correlation condition (" << name << ")" << std::endl;

    return false;
  }

  // Define some of the quantities to store the parased information

  // condition type BLW  (Do we change this to the type of correlation condition?)
  GtConditionType cType = l1t::Type2cor;

  // two objects (for sure)
  const int nrObj = 2;

  // object types and greater equal flag - filled in the loop
  int intGEq[nrObj] = {-1, -1};
  std::vector<GlobalObject> objType(nrObj);           //BLW do we want to define these as a different type?
  std::vector<GtConditionCategory> condCateg(nrObj);  //BLW do we want to change these categories

  // correlation flag and index in the cor*vector
  const bool corrFlag = true;
  int corrIndexVal[nrObj] = {-1, -1};

  // Storage of the correlation selection
  CorrelationTemplate::CorrelationParameter corrParameter;
  corrParameter.chargeCorrelation = 1;  //ignore charge correlation

  // Get the correlation Cuts on the legs
  int cutType = 0;
  const std::vector<L1TUtmCut>& cuts = corrCond.getCuts();
  for (size_t jj = 0; jj < cuts.size(); jj++) {
    const L1TUtmCut& cut = cuts.at(jj);

    if (cut.getCutType() == esCutType::ChargeCorrelation) {
      if (cut.getData() == "ls")
        corrParameter.chargeCorrelation = 2;
      else if (cut.getData() == "os")
        corrParameter.chargeCorrelation = 4;
      else
        corrParameter.chargeCorrelation = 1;  //ignore charge correlation
    } else {
      //
      //  Until utm has method to calculate these, do the integer value calculation with precision.
      //
      double minV = cut.getMinimum().value;
      double maxV = cut.getMaximum().value;

      //Scale down very large numbers out of xml
      if (maxV > 1.0e8)
        maxV = 1.0e8;

      if (cut.getCutType() == esCutType::DeltaEta) {
        LogDebug("TriggerMenuParser") << "CutType: " << cut.getCutType() << "\tDeltaEta Cut minV = " << minV
                                      << " Max = " << maxV << " precMin = " << cut.getMinimum().index
                                      << " precMax = " << cut.getMaximum().index << std::endl;
        corrParameter.minEtaCutValue = (long long)(minV * pow(10., cut.getMinimum().index));
        corrParameter.maxEtaCutValue = (long long)(maxV * pow(10., cut.getMaximum().index));
        corrParameter.precEtaCut = cut.getMinimum().index;
        cutType = cutType | 0x1;
      } else if (cut.getCutType() == esCutType::DeltaPhi) {
        LogDebug("TriggerMenuParser") << "CutType: " << cut.getCutType() << "\tDeltaPhi Cut minV = " << minV
                                      << " Max = " << maxV << " precMin = " << cut.getMinimum().index
                                      << " precMax = " << cut.getMaximum().index << std::endl;
        corrParameter.minPhiCutValue = (long long)(minV * pow(10., cut.getMinimum().index));
        corrParameter.maxPhiCutValue = (long long)(maxV * pow(10., cut.getMaximum().index));
        corrParameter.precPhiCut = cut.getMinimum().index;
        cutType = cutType | 0x2;
      } else if (cut.getCutType() == esCutType::DeltaR) {
        LogDebug("TriggerMenuParser") << "CutType: " << cut.getCutType() << "\tDeltaR Cut minV = " << minV
                                      << " Max = " << maxV << " precMin = " << cut.getMinimum().index
                                      << " precMax = " << cut.getMaximum().index << std::endl;
        corrParameter.minDRCutValue = (long long)(minV * pow(10., cut.getMinimum().index));
        corrParameter.maxDRCutValue = (long long)(maxV * pow(10., cut.getMaximum().index));
        corrParameter.precDRCut = cut.getMinimum().index;
        cutType = cutType | 0x4;
      } else if (cut.getCutType() == esCutType::TwoBodyPt) {
        corrParameter.minTBPTCutValue = (long long)(minV * pow(10., cut.getMinimum().index));
        corrParameter.maxTBPTCutValue = (long long)(maxV * pow(10., cut.getMaximum().index));
        corrParameter.precTBPTCut = cut.getMinimum().index;
        LogDebug("TriggerMenuParser") << "CutType: " << cut.getCutType() << "\tTPBT Cut minV = " << minV
                                      << " Max = " << maxV << " precMin = " << cut.getMinimum().index
                                      << " precMax = " << cut.getMaximum().index << std::endl;
        cutType = cutType | 0x20;
      } else if ((cut.getCutType() == esCutType::Mass) ||
                 (cut.getCutType() == esCutType::MassDeltaR)) {  //Invariant Mass, MassOverDeltaR
        LogDebug("TriggerMenuParser") << "CutType: " << cut.getCutType() << "\tMass Cut minV = " << minV
                                      << " Max = " << maxV << " precMin = " << cut.getMinimum().index
                                      << " precMax = " << cut.getMaximum().index << std::endl;
        corrParameter.minMassCutValue = (long long)(minV * pow(10., cut.getMinimum().index));
        corrParameter.maxMassCutValue = (long long)(maxV * pow(10., cut.getMaximum().index));
        corrParameter.precMassCut = cut.getMinimum().index;
        // cutType = cutType | 0x8;
        if (corrCond.getType() == esConditionType::TransverseMass) {
          cutType = cutType | 0x10;
        } else if (corrCond.getType() == esConditionType::InvariantMassDeltaR) {
          cutType = cutType | 0x80;
        } else {
          cutType = cutType | 0x8;
        }
      } else if (cut.getCutType() == esCutType::MassUpt) {  // Added for displaced muons
        LogDebug("TriggerMenuParser") << "CutType: " << cut.getCutType() << "\tMass Cut minV = " << minV
                                      << " Max = " << maxV << " precMin = " << cut.getMinimum().index
                                      << " precMax = " << cut.getMaximum().index << std::endl;
        corrParameter.minMassCutValue = (long long)(minV * pow(10., cut.getMinimum().index));
        corrParameter.maxMassCutValue = (long long)(maxV * pow(10., cut.getMaximum().index));
        corrParameter.precMassCut = cut.getMinimum().index;
        cutType = cutType | 0x40;  // Note:    0x40 (MassUpt) is next available bit after 0x20 (TwoBodyPt)
      }                            // Careful: cutType carries same info as esCutType, but is hard coded!!
    }                              //          This seems like a historical hack, which may be error prone.
  }                                //          cutType is defined here, for use later in CorrCondition.cc
  corrParameter.corrCutType = cutType;

  // Get the two objects that form the legs
  const std::vector<L1TUtmObject>& objects = corrCond.getObjects();
  if (objects.size() != 2) {
    edm::LogError("TriggerMenuParser") << "incorrect number of objects for the correlation condition " << name
                                       << " corrFlag " << corrFlag << std::endl;
    return false;
  }

  // loop over legs
  for (size_t jj = 0; jj < objects.size(); jj++) {
    const L1TUtmObject& object = objects.at(jj);
    LogDebug("TriggerMenuParser") << "      obj name = " << object.getName() << "\n";
    LogDebug("TriggerMenuParser") << "      obj type = " << object.getType() << "\n";
    LogDebug("TriggerMenuParser") << "      obj op = " << object.getComparisonOperator() << "\n";
    LogDebug("TriggerMenuParser") << "      obj bx = " << object.getBxOffset() << "\n";

    // check the leg type
    if (object.getType() == esObjectType::Muon) {
      // we have a muon

      /*
          //BLW Hold on to this code we may need to go back to it at some point.
	  // Now we are putting ALL leg conditions into the vector (so there are duplicates)
	  // This is potentially a place to slim down the code.  Note: We currently evaluate the
	  // conditions every time, so even if we put the condition in the vector once, we would
	  // still evaluate it multiple times.  This is a place for optimization.
          {

              parseMuonCorr(&object,chipNr);
	      corrIndexVal[jj] = (m_corMuonTemplate[chipNr]).size() - 1;

          } else {
	     LogDebug("TriggerMenuParser") << "Not Adding Correlation Muon Condition to Map...looking for the condition in Muon Cor Vector" << std::endl;
	     bool found = false;
	     int index = 0;
	     while(!found && index<(int)((m_corMuonTemplate[chipNr]).size()) ) {
	         if( (m_corMuonTemplate[chipNr]).at(index).condName() == object.getName() ) {
		    LogDebug("TriggerMenuParser") << "Found condition " << object.getName() << " in vector at index " << index << std::endl;
		    found = true;
		 } else {
		    index++;
		 }
	     }
	     if(found) {
	        corrIndexVal[jj] = index;
	     } else {
	       edm::LogError("TriggerMenuParser") << "FAILURE: Condition " << object.getName() << " is in map but not in cor. vector " << std::endl;
	     }

	  }
*/
      parseMuonCorr(&object, chipNr);
      corrIndexVal[jj] = (m_corMuonTemplate[chipNr]).size() - 1;

      //Now set some flags for this subCondition
      intGEq[jj] = (object.getComparisonOperator() == esComparisonOperator::GE);
      objType[jj] = gtMu;
      condCateg[jj] = CondMuon;

    } else if (object.getType() == esObjectType::Egamma || object.getType() == esObjectType::Jet ||
               object.getType() == esObjectType::Tau) {
      // we have an Calo object
      parseCaloCorr(&object, chipNr);
      corrIndexVal[jj] = (m_corCaloTemplate[chipNr]).size() - 1;

      //Now set some flags for this subCondition
      intGEq[jj] = (object.getComparisonOperator() == esComparisonOperator::GE);
      switch (object.getType()) {
        case esObjectType::Egamma: {
          objType[jj] = gtEG;
        } break;
        case esObjectType::Jet: {
          objType[jj] = gtJet;
        } break;
        case esObjectType::Tau: {
          objType[jj] = gtTau;
        } break;
        default: {
        } break;
      }
      condCateg[jj] = CondCalo;

    } else if (object.getType() == esObjectType::ETM || object.getType() == esObjectType::ETMHF ||
               object.getType() == esObjectType::TOWERCOUNT || object.getType() == esObjectType::HTM) {
      // we have Energy Sum
      parseEnergySumCorr(&object, chipNr);
      corrIndexVal[jj] = (m_corEnergySumTemplate[chipNr]).size() - 1;

      //Now set some flags for this subCondition
      intGEq[jj] = (object.getComparisonOperator() == esComparisonOperator::GE);
      switch (object.getType()) {
        case esObjectType::ETM: {
          objType[jj] = GlobalObject::gtETM;
        } break;
        case esObjectType::HTM: {
          objType[jj] = GlobalObject::gtHTM;
        } break;
        case esObjectType::ETMHF: {
          objType[jj] = GlobalObject::gtETMHF;
        } break;
        case esObjectType::TOWERCOUNT: {
          objType[jj] = GlobalObject::gtTowerCount;
        } break;
        default: {
        } break;
      }
      condCateg[jj] = CondEnergySum;

    } else {
      edm::LogError("TriggerMenuParser") << "Illegal Object Type " << object.getType()
                                         << " for the correlation condition " << name << std::endl;
      return false;

    }  //if block on leg types

  }  //loop over legs

  // get greater equal flag for the correlation condition
  bool gEq = true;
  if (intGEq[0] != intGEq[1]) {
    edm::LogError("TriggerMenuParser") << "Inconsistent GEq flags for sub-conditions "
                                       << " for the correlation condition " << name << std::endl;
    return false;

  } else {
    gEq = (intGEq[0] != 0);
  }

  // fill the correlation condition
  correlationCond.setCondType(cType);
  correlationCond.setObjectType(objType);
  correlationCond.setCondGEq(gEq);
  correlationCond.setCondChipNr(chipNr);

  correlationCond.setCond0Category(condCateg[0]);
  correlationCond.setCond1Category(condCateg[1]);

  correlationCond.setCond0Index(corrIndexVal[0]);
  correlationCond.setCond1Index(corrIndexVal[1]);

  correlationCond.setCorrelationParameter(corrParameter);

  if (edm::isDebugEnabled()) {
    std::ostringstream myCoutStream;
    correlationCond.print(myCoutStream);
    LogTrace("TriggerMenuParser") << myCoutStream.str() << "\n" << std::endl;
  }

  // insert condition into the map
  // condition is not duplicate, check was done at the beginning

  (m_vecCorrelationTemplate[chipNr]).push_back(correlationCond);

  //
  return true;
}

//////////////////////////////////////////////////////////////////////////////////////
/**
 * parseCorrelationThreeBody Parse a correlation condition between three objects and
 * insert an entry to the conditions map
 *
 * @param node The corresponding node.
 * @param name The name of the condition.
 * @param chipNr The number of the chip this condition is located.
 *
 * @return "true" if succeeded, "false" if an error occurred.
 *
 */

bool l1t::TriggerMenuParser::parseCorrelationThreeBody(L1TUtmCondition corrCond, unsigned int chipNr) {
  using namespace tmeventsetup;
  std::string condition = "corrThreeBody";
  std::string particle = "muon";
  std::string type = l1t2string(corrCond.getType());
  std::string name = l1t2string(corrCond.getName());

  LogDebug("TriggerMenuParser") << " ****************************************** " << std::endl
                                << "     (in parseCorrelationThreeBody) " << std::endl
                                << " condition = " << condition << std::endl
                                << " particle  = " << particle << std::endl
                                << " type      = " << type << std::endl
                                << " name      = " << name << std::endl;

  // create a new correlation condition
  CorrelationThreeBodyTemplate correlationThreeBodyCond(name);

  // check that the condition does not exist already in the map
  if (!insertConditionIntoMap(correlationThreeBodyCond, chipNr)) {
    edm::LogError("TriggerMenuParser") << "    Error: duplicate correlation condition (" << name << ")" << std::endl;
    return false;
  }

  // Define some of the quantities to store the parsed information
  GtConditionType cType = l1t::Type3s;

  // three objects (for sure)
  const int nrObj = 3;

  // object types and greater equal flag - filled in the loop
  std::vector<GlobalObject> objType(nrObj);
  std::vector<GtConditionCategory> condCateg(nrObj);

  // correlation flag and index in the cor*vector
  const bool corrFlag = true;
  int corrIndexVal[nrObj] = {-1, -1, -1};

  // Storage of the correlation selection
  CorrelationThreeBodyTemplate::CorrelationThreeBodyParameter corrThreeBodyParameter;
  // Set charge correlation parameter
  //corrThreeBodyParameter.chargeCorrelation = chargeCorrelation;  //tmpValues[0];
  //corrThreeBodyParameter.chargeCorrelation = 1;  //ignore charge correlation for corr-legs

  // Get the correlation cuts on the legs
  int cutType = 0;
  const std::vector<L1TUtmCut>& cuts = corrCond.getCuts();
  for (size_t lll = 0; lll < cuts.size(); lll++) {  // START esCut lll
    const L1TUtmCut& cut = cuts.at(lll);

    if (cut.getCutType() == esCutType::ChargeCorrelation) {
      if (cut.getData() == "ls")
        corrThreeBodyParameter.chargeCorrelation = 2;
      else if (cut.getData() == "os")
        corrThreeBodyParameter.chargeCorrelation = 4;
      else
        corrThreeBodyParameter.chargeCorrelation = 1;  //ignore charge correlation
    }

    //
    //  Until utm has method to calculate these, do the integer value calculation with precision.
    //
    double minV = cut.getMinimum().value;
    double maxV = cut.getMaximum().value;
    //Scale down very large numbers out of xml
    if (maxV > 1.0e8)
      maxV = 1.0e8;

    if (cut.getCutType() == esCutType::Mass) {
      LogDebug("TriggerMenuParser") << "CutType: " << cut.getCutType() << "\tMass Cut minV = " << minV
                                    << "\tMass Cut maxV = " << maxV << " precMin = " << cut.getMinimum().index
                                    << " precMax = " << cut.getMaximum().index << std::endl;
      corrThreeBodyParameter.minMassCutValue = (long long)(minV * pow(10., cut.getMinimum().index));
      corrThreeBodyParameter.maxMassCutValue = (long long)(maxV * pow(10., cut.getMaximum().index));
      corrThreeBodyParameter.precMassCut = cut.getMinimum().index;
      cutType = cutType | 0x8;
    } else if (cut.getCutType() == esCutType::MassDeltaR) {
      corrThreeBodyParameter.minMassCutValue = (long long)(minV * pow(10., cut.getMinimum().index));
      corrThreeBodyParameter.maxMassCutValue = (long long)(maxV * pow(10., cut.getMaximum().index));
      corrThreeBodyParameter.precMassCut = cut.getMinimum().index;
      cutType = cutType | 0x80;
    }
  }  // END esCut lll
  corrThreeBodyParameter.corrCutType = cutType;

  // Get the three objects that form the legs
  const std::vector<L1TUtmObject>& objects = corrCond.getObjects();
  if (objects.size() != 3) {
    edm::LogError("TriggerMenuParser") << "incorrect number of objects for the correlation condition " << name
                                       << " corrFlag " << corrFlag << std::endl;
    return false;
  }

  // Loop over legs
  for (size_t lll = 0; lll < objects.size(); lll++) {
    const L1TUtmObject& object = objects.at(lll);
    LogDebug("TriggerMenuParser") << "      obj name = " << object.getName() << "\n";
    LogDebug("TriggerMenuParser") << "      obj type = " << object.getType() << "\n";
    LogDebug("TriggerMenuParser") << "      obj bx = " << object.getBxOffset() << "\n";

    // check the leg type
    if (object.getType() == esObjectType::Muon) {
      // we have a muon
      parseMuonCorr(&object, chipNr);
      corrIndexVal[lll] = (m_corMuonTemplate[chipNr]).size() - 1;

      //Now set some flags for this subCondition
      objType[lll] = gtMu;
      condCateg[lll] = CondMuon;

    } else {
      edm::LogError("TriggerMenuParser") << "Checked the object Type " << object.getType()
                                         << " for the correlation condition " << name
                                         << ": no three muons in the event!" << std::endl;
    }
  }  // End loop over legs

  // fill the three-body correlation condition
  correlationThreeBodyCond.setCondType(cType);
  correlationThreeBodyCond.setObjectType(objType);
  correlationThreeBodyCond.setCondChipNr(chipNr);

  correlationThreeBodyCond.setCond0Category(condCateg[0]);
  correlationThreeBodyCond.setCond1Category(condCateg[1]);
  correlationThreeBodyCond.setCond2Category(condCateg[2]);

  correlationThreeBodyCond.setCond0Index(corrIndexVal[0]);
  correlationThreeBodyCond.setCond1Index(corrIndexVal[1]);
  correlationThreeBodyCond.setCond2Index(corrIndexVal[2]);

  correlationThreeBodyCond.setCorrelationThreeBodyParameter(corrThreeBodyParameter);

  if (edm::isDebugEnabled()) {
    std::ostringstream myCoutStream;
    correlationThreeBodyCond.print(myCoutStream);
    LogTrace("TriggerMenuParser") << myCoutStream.str() << "\n" << std::endl;
  }

  // insert condition into the map
  // condition is not duplicate, check was done at the beginning

  (m_vecCorrelationThreeBodyTemplate[chipNr]).push_back(correlationThreeBodyCond);

  //
  return true;
}

////////////////////////////////////////////////////////////////////////////////////
/**
 * parseCorrelationWithOverlapRemoval Parse a correlation condition and
 * insert an entry to the conditions map
 *
 * @param node The corresponding node.
 * @param name The name of the condition.
 * @param chipNr The number of the chip this condition is located.
 *
 * @return "true" if succeeded, "false" if an error occurred.
 *
 */

bool l1t::TriggerMenuParser::parseCorrelationWithOverlapRemoval(const L1TUtmCondition& corrCond, unsigned int chipNr) {
  using namespace tmeventsetup;
  std::string condition = "corrWithOverlapRemoval";
  std::string particle = "test-fix";
  std::string type = l1t2string(corrCond.getType());
  std::string name = l1t2string(corrCond.getName());

  LogDebug("TriggerMenuParser") << " ****************************************** " << std::endl
                                << "     (in parseCorrelationWithOverlapRemoval) " << std::endl
                                << " condition = " << condition << std::endl
                                << " particle  = " << particle << std::endl
                                << " type      = " << type << std::endl
                                << " name      = " << name << std::endl;

  // create a new correlation condition
  CorrelationWithOverlapRemovalTemplate correlationWORCond(name);

  // check that the condition does not exist already in the map
  if (!insertConditionIntoMap(correlationWORCond, chipNr)) {
    edm::LogError("TriggerMenuParser") << "    Error: duplicate correlation condition (" << name << ")" << std::endl;

    return false;
  }

  // Define some of the quantities to store the parased information

  // condition type BLW  (Do we change this to the type of correlation condition?)
  GtConditionType cType = l1t::Type2corWithOverlapRemoval;

  // three objects (for sure)
  const int nrObj = 3;

  // object types and greater equal flag - filled in the loop
  int intGEq[nrObj] = {-1, -1, -1};
  std::vector<GlobalObject> objType(nrObj);           //BLW do we want to define these as a different type?
  std::vector<GtConditionCategory> condCateg(nrObj);  //BLW do we want to change these categories

  // correlation flag and index in the cor*vector
  const bool corrFlag = true;
  int corrIndexVal[nrObj] = {-1, -1, -1};

  // Storage of the correlation selection
  CorrelationWithOverlapRemovalTemplate::CorrelationWithOverlapRemovalParameter corrParameter;
  corrParameter.chargeCorrelation = 1;  //ignore charge correlation for corr-legs

  // Get the correlation Cuts on the legs
  int cutType = 0;
  const std::vector<L1TUtmCut>& cuts = corrCond.getCuts();
  for (size_t jj = 0; jj < cuts.size(); jj++) {
    const L1TUtmCut& cut = cuts.at(jj);

    if (cut.getCutType() == esCutType::ChargeCorrelation) {
      if (cut.getData() == "ls")
        corrParameter.chargeCorrelation = 2;
      else if (cut.getData() == "os")
        corrParameter.chargeCorrelation = 4;
      else
        corrParameter.chargeCorrelation = 1;  //ignore charge correlation
    } else {
      //
      //  Unitl utm has method to calculate these, do the integer value calculation with precision.
      //
      double minV = cut.getMinimum().value;
      double maxV = cut.getMaximum().value;

      //Scale down very large numbers out of xml
      if (maxV > 1.0e8)
        maxV = 1.0e8;

      if (cut.getCutType() == esCutType::DeltaEta) {
        //std::cout << "DeltaEta Cut minV = " << minV << " Max = " << maxV << " precMin = " << cut.getMinimum().index << " precMax = " << cut.getMaximum().index << std::endl;
        corrParameter.minEtaCutValue = (long long)(minV * pow(10., cut.getMinimum().index));
        corrParameter.maxEtaCutValue = (long long)(maxV * pow(10., cut.getMaximum().index));
        corrParameter.precEtaCut = cut.getMinimum().index;
        cutType = cutType | 0x1;
      } else if (cut.getCutType() == esCutType::DeltaPhi) {
        //std::cout << "DeltaPhi Cut minV = " << minV << " Max = " << maxV << " precMin = " << cut.getMinimum().index << " precMax = " << cut.getMaximum().index << std::endl;
        corrParameter.minPhiCutValue = (long long)(minV * pow(10., cut.getMinimum().index));
        corrParameter.maxPhiCutValue = (long long)(maxV * pow(10., cut.getMaximum().index));
        corrParameter.precPhiCut = cut.getMinimum().index;
        cutType = cutType | 0x2;
      } else if (cut.getCutType() == esCutType::DeltaR) {
        //std::cout << "DeltaR Cut minV = " << minV << " Max = " << maxV << " precMin = " << cut.getMinimum().index << " precMax = " << cut.getMaximum().index << std::endl;
        corrParameter.minDRCutValue = (long long)(minV * pow(10., cut.getMinimum().index));
        corrParameter.maxDRCutValue = (long long)(maxV * pow(10., cut.getMaximum().index));
        corrParameter.precDRCut = cut.getMinimum().index;
        cutType = cutType | 0x4;
      } else if (cut.getCutType() == esCutType::Mass) {
        //std::cout << "Mass Cut minV = " << minV << " Max = " << maxV << " precMin = " << cut.getMinimum().index << " precMax = " << cut.getMaximum().index << std::endl;
        corrParameter.minMassCutValue = (long long)(minV * pow(10., cut.getMinimum().index));
        corrParameter.maxMassCutValue = (long long)(maxV * pow(10., cut.getMaximum().index));
        corrParameter.precMassCut = cut.getMinimum().index;
        cutType = cutType | 0x8;
      } else if (cut.getCutType() == esCutType::MassDeltaR) {
        corrParameter.minMassCutValue = (long long)(minV * pow(10., cut.getMinimum().index));
        corrParameter.maxMassCutValue = (long long)(maxV * pow(10., cut.getMaximum().index));
        corrParameter.precMassCut = cut.getMinimum().index;
        cutType = cutType | 0x80;
      }
      if (cut.getCutType() == esCutType::OvRmDeltaEta) {
        //std::cout << "OverlapRemovalDeltaEta Cut minV = " << minV << " Max = " << maxV << " precMin = " << cut.getMinimum().index << " precMax = " << cut.getMaximum().index << std::endl;
        corrParameter.minOverlapRemovalEtaCutValue = (long long)(minV * pow(10., cut.getMinimum().index));
        corrParameter.maxOverlapRemovalEtaCutValue = (long long)(maxV * pow(10., cut.getMaximum().index));
        corrParameter.precOverlapRemovalEtaCut = cut.getMinimum().index;
        cutType = cutType | 0x10;
      } else if (cut.getCutType() == esCutType::OvRmDeltaPhi) {
        //std::cout << "OverlapRemovalDeltaPhi Cut minV = " << minV << " Max = " << maxV << " precMin = " << cut.getMinimum().index << " precMax = " << cut.getMaximum().index << std::endl;
        corrParameter.minOverlapRemovalPhiCutValue = (long long)(minV * pow(10., cut.getMinimum().index));
        corrParameter.maxOverlapRemovalPhiCutValue = (long long)(maxV * pow(10., cut.getMaximum().index));
        corrParameter.precOverlapRemovalPhiCut = cut.getMinimum().index;
        cutType = cutType | 0x20;
      } else if (cut.getCutType() == esCutType::OvRmDeltaR) {
        //std::cout << "DeltaR Cut minV = " << minV << " Max = " << maxV << " precMin = " << cut.getMinimum().index << " precMax = " << cut.getMaximum().index << std::endl;
        corrParameter.minOverlapRemovalDRCutValue = (long long)(minV * pow(10., cut.getMinimum().index));
        corrParameter.maxOverlapRemovalDRCutValue = (long long)(maxV * pow(10., cut.getMaximum().index));
        corrParameter.precOverlapRemovalDRCut = cut.getMinimum().index;
        cutType = cutType | 0x40;
      }
    }
  }
  corrParameter.corrCutType = cutType;

  // Get the two objects that form the legs
  const std::vector<L1TUtmObject>& objects = corrCond.getObjects();
  if (objects.size() != 3) {
    edm::LogError("TriggerMenuParser")
        << "incorrect number of objects for the correlation condition with overlap removal " << name << " corrFlag "
        << corrFlag << std::endl;
    return false;
  }

  // Loop over legs
  for (size_t jj = 0; jj < objects.size(); jj++) {
    const L1TUtmObject& object = objects.at(jj);
    LogDebug("TriggerMenuParser") << "      obj name = " << object.getName() << "\n";
    LogDebug("TriggerMenuParser") << "      obj type = " << object.getType() << "\n";
    LogDebug("TriggerMenuParser") << "      obj op = " << object.getComparisonOperator() << "\n";
    LogDebug("TriggerMenuParser") << "      obj bx = " << object.getBxOffset() << "\n";
    LogDebug("TriggerMenuParser") << "type = done" << std::endl;

    // check the leg type
    if (object.getType() == esObjectType::Muon) {
      // we have a muon

      /*
          //BLW Hold on to this code we may need to go back to it at some point.
	  // Now we are putting ALL leg conditions into the vector (so there are duplicates)
	  // This is potentially a place to slim down the code.  Note: We currently evaluate the
	  // conditions every time, so even if we put the condition in the vector once, we would
	  // still evaluate it multiple times.  This is a place for optimization.
          {

              parseMuonCorr(&object,chipNr);
	      corrIndexVal[jj] = (m_corMuonTemplate[chipNr]).size() - 1;

          } else {
	     LogDebug("TriggerMenuParser") << "Not Adding Correlation Muon Condition to Map...looking for the condition in Muon Cor Vector" << std::endl;
	     bool found = false;
	     int index = 0;
	     while(!found && index<(int)((m_corMuonTemplate[chipNr]).size()) ) {
	         if( (m_corMuonTemplate[chipNr]).at(index).condName() == object.getName() ) {
		    LogDebug("TriggerMenuParser") << "Found condition " << object.getName() << " in vector at index " << index << std::endl;
		    found = true;
		 } else {
		    index++;
		 }
	     }
	     if(found) {
	        corrIndexVal[jj] = index;
	     } else {
	       edm::LogError("TriggerMenuParser") << "FAILURE: Condition " << object.getName() << " is in map but not in cor. vector " << std::endl;
	     }

	  }
*/
      parseMuonCorr(&object, chipNr);
      corrIndexVal[jj] = (m_corMuonTemplate[chipNr]).size() - 1;

      //Now set some flags for this subCondition
      intGEq[jj] = (object.getComparisonOperator() == esComparisonOperator::GE);
      objType[jj] = gtMu;
      condCateg[jj] = CondMuon;

    } else if (object.getType() == esObjectType::Egamma || object.getType() == esObjectType::Jet ||
               object.getType() == esObjectType::Tau) {
      // we have an Calo object
      parseCaloCorr(&object, chipNr);
      corrIndexVal[jj] = (m_corCaloTemplate[chipNr]).size() - 1;

      //Now set some flags for this subCondition
      intGEq[jj] = (object.getComparisonOperator() == esComparisonOperator::GE);
      switch (object.getType()) {
        case esObjectType::Egamma: {
          objType[jj] = gtEG;
        } break;
        case esObjectType::Jet: {
          objType[jj] = gtJet;
        } break;
        case esObjectType::Tau: {
          objType[jj] = gtTau;
        } break;
        default: {
        } break;
      }
      condCateg[jj] = CondCalo;

    } else if (object.getType() == esObjectType::ETM || object.getType() == esObjectType::ETMHF ||
               object.getType() == esObjectType::TOWERCOUNT || object.getType() == esObjectType::HTM) {
      // we have Energy Sum
      parseEnergySumCorr(&object, chipNr);
      corrIndexVal[jj] = (m_corEnergySumTemplate[chipNr]).size() - 1;

      //Now set some flags for this subCondition
      intGEq[jj] = (object.getComparisonOperator() == esComparisonOperator::GE);
      switch (object.getType()) {
        case esObjectType::ETM: {
          objType[jj] = GlobalObject::gtETM;
        } break;
        case esObjectType::HTM: {
          objType[jj] = GlobalObject::gtHTM;
        } break;
        case esObjectType::ETMHF: {
          objType[jj] = GlobalObject::gtETMHF;
        } break;
        case esObjectType::TOWERCOUNT: {
          objType[jj] = GlobalObject::gtTowerCount;
        } break;
        default: {
        } break;
      }
      condCateg[jj] = CondEnergySum;

    } else {
      edm::LogError("TriggerMenuParser") << "Illegal Object Type " << object.getType()
                                         << " for the correlation condition " << name << std::endl;
      return false;

    }  //if block on leg types

  }  //loop over legs

  // get greater equal flag for the correlation condition
  bool gEq = true;
  if (intGEq[0] != intGEq[1]) {
    edm::LogError("TriggerMenuParser") << "Inconsistent GEq flags for sub-conditions "
                                       << " for the correlation condition " << name << std::endl;
    return false;

  } else {
    gEq = (intGEq[0] != 0);
  }

  // fill the correlation condition
  correlationWORCond.setCondType(cType);
  correlationWORCond.setObjectType(objType);
  correlationWORCond.setCondGEq(gEq);
  correlationWORCond.setCondChipNr(chipNr);

  correlationWORCond.setCond0Category(condCateg[0]);
  correlationWORCond.setCond1Category(condCateg[1]);
  correlationWORCond.setCond2Category(condCateg[2]);

  correlationWORCond.setCond0Index(corrIndexVal[0]);
  correlationWORCond.setCond1Index(corrIndexVal[1]);
  correlationWORCond.setCond2Index(corrIndexVal[2]);

  correlationWORCond.setCorrelationWithOverlapRemovalParameter(corrParameter);

  if (edm::isDebugEnabled()) {
    std::ostringstream myCoutStream;
    correlationWORCond.print(myCoutStream);
    LogTrace("TriggerMenuParser") << myCoutStream.str() << "\n" << std::endl;
  }

  // insert condition into the map
  // condition is not duplicate, check was done at the beginning

  (m_vecCorrelationWithOverlapRemovalTemplate[chipNr]).push_back(correlationWORCond);

  //
  return true;
}

/**
 * workAlgorithm - parse the algorithm and insert it into algorithm map.
 *
 * @param node The corresponding node to the algorithm.
 * @param name The name of the algorithm.
 * @param chipNr The number of the chip the conditions for that algorithm are located on.
 *
 * @return "true" on success, "false" if an error occurred.
 *
 */

bool l1t::TriggerMenuParser::parseAlgorithm(L1TUtmAlgorithm algorithm, unsigned int chipNr) {
  // get alias
  std::string algAlias = algorithm.getName();
  const std::string& algName = algorithm.getName();

  if (algAlias.empty()) {
    algAlias = algName;
    LogDebug("TriggerMenuParser") << "\n    No alias defined for algorithm. Alias set to algorithm name."
                                  << "\n    Algorithm name:  " << algName << "\n    Algorithm alias: " << algAlias
                                  << std::endl;
  } else {
    //LogDebug("TriggerMenuParser")
    LogDebug("TriggerMenuParser") << "\n    Alias defined for algorithm."
                                  << "\n    Algorithm name:  " << algName << "\n    Algorithm alias: " << algAlias
                                  << std::endl;
  }

  // get the logical expression
  const std::string& logExpression = algorithm.getExpressionInCondition();

  LogDebug("TriggerMenuParser") << "      Logical expression: " << logExpression
                                << "      Chip number:        " << chipNr << std::endl;

  // determine output pin
  int outputPin = algorithm.getIndex();

  //LogTrace("TriggerMenuParser")
  LogDebug("TriggerMenuParser") << "      Output pin:         " << outputPin << std::endl;

  // compute the bit number from chip number, output pin and order of the chips
  // pin numbering start with 1, bit numbers with 0
  int bitNumber = outputPin;  // + (m_orderConditionChip[chipNr] -1)*m_pinsOnConditionChip -1;

  //LogTrace("TriggerMenuParser")
  LogDebug("TriggerMenuParser") << "      Bit number:         " << bitNumber << std::endl;

  // create a new algorithm and insert it into algorithm map
  GlobalAlgorithm alg(algName, logExpression, bitNumber);
  alg.setAlgoChipNumber(static_cast<int>(chipNr));
  alg.setAlgoAlias(algAlias);

  if (edm::isDebugEnabled()) {
    std::ostringstream myCoutStream;
    alg.print(myCoutStream);
    LogTrace("TriggerMenuParser") << myCoutStream.str() << "\n" << std::endl;
  }

  // insert algorithm into the map
  if (!insertAlgorithmIntoMap(alg)) {
    return false;
  }

  return true;
}
// static class members

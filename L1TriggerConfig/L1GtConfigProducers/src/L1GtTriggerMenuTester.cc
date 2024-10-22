/**
 * \class L1GtTriggerMenuTester
 *
 *
 * Description: test analyzer for L1 GT trigger menu.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
 *
 */

// this class header
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtTriggerMenuTester.h"

// system include files
#include <iomanip>
#include <boost/algorithm/string/erase.hpp>

// user include files
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1GtStableParameters.h"
#include "CondFormats/DataRecord/interface/L1GtStableParametersRcd.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"

#include "CondFormats/L1TObjects/interface/L1GtPrescaleFactors.h"

#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsTechTrigRcd.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"

#include "CondFormats/DataRecord/interface/L1GtTriggerMaskAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskTechTrigRcd.h"

#include "CondFormats/DataRecord/interface/L1GtTriggerMaskVetoAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskVetoTechTrigRcd.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GtLogicParser.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

// forward declarations

// constructor(s)
L1GtTriggerMenuTester::L1GtTriggerMenuTester(const edm::ParameterSet& parSet)
    : m_overwriteHtmlFile(parSet.getParameter<bool>("OverwriteHtmlFile")),
      m_htmlFile(parSet.getParameter<std::string>("HtmlFile")),
      m_useHltMenu(parSet.getParameter<bool>("UseHltMenu")),
      m_hltProcessName(parSet.getParameter<std::string>("HltProcessName")),
      m_noThrowIncompatibleMenu(parSet.getParameter<bool>("NoThrowIncompatibleMenu")),
      m_printPfsRates(parSet.getParameter<bool>("PrintPfsRates")),
      m_indexPfSet(parSet.getParameter<int>("IndexPfSet")),
      m_l1GtStableParToken(esConsumes<edm::Transition::BeginRun>()),
      m_l1GtPfAlgoToken(esConsumes<edm::Transition::BeginRun>()),
      m_l1GtPfTechToken(esConsumes<edm::Transition::BeginRun>()),
      m_l1GtTmTechToken(esConsumes<edm::Transition::BeginRun>()),
      m_l1GtTmVetoAlgoToken(esConsumes<edm::Transition::BeginRun>()),
      m_l1GtTmVetoTechToken(esConsumes<edm::Transition::BeginRun>()),
      m_l1GtMenuToken(esConsumes<edm::Transition::BeginRun>()),
      m_numberAlgorithmTriggers(0),
      m_numberTechnicalTriggers(0) {
  // empty
}

// begin run
void L1GtTriggerMenuTester::beginRun(const edm::Run& iRun, const edm::EventSetup& evSetup) {
  // retrieve L1 trigger configuration
  retrieveL1EventSetup(evSetup);

  // print with various level of verbosity

  // define an output stream to print into
  // it can then be directed to whatever log level is desired
  std::ostringstream myCout;

  int printVerbosity = 0;
  m_l1GtMenu->print(myCout, printVerbosity);
  myCout << std::flush << std::endl;

  printVerbosity = 1;
  m_l1GtMenu->print(myCout, printVerbosity);
  myCout << std::flush << std::endl;

  printVerbosity = 2;
  m_l1GtMenu->print(myCout, printVerbosity);
  myCout << std::flush << std::endl;

  // redirect myCout to edm::LogVerbatim TODO - parameter to choose the log
  edm::LogVerbatim("L1GtTriggerMenuTester") << myCout.str() << std::endl;

  // prepare L1 - HLT
  if (m_useHltMenu) {
    associateL1SeedsHltPath(iRun, evSetup);

    if (m_noThrowIncompatibleMenu) {
      edm::LogVerbatim("L1GtTriggerMenuTester")
          << "\n List of algorithm triggers used as L1 seeds but not in L1 menu" << std::endl;

      for (std::vector<std::string>::const_iterator strIter = m_algoTriggerSeedNotInL1Menu.begin();
           strIter != m_algoTriggerSeedNotInL1Menu.end();
           ++strIter) {
        edm::LogVerbatim("L1GtTriggerMenuTester") << "   " << (*strIter) << std::endl;
      }
    }
  }

  // print in wiki format
  printWiki();
}

// loop over events
void L1GtTriggerMenuTester::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  // empty
}

// end run
void L1GtTriggerMenuTester::endRun(const edm::Run&, const edm::EventSetup& evSetup) {}

void L1GtTriggerMenuTester::retrieveL1EventSetup(const edm::EventSetup& evSetup) {
  // get / update the stable parameters from the EventSetup

  m_l1GtStablePar = &evSetup.getData(m_l1GtStableParToken);

  // number of algorithm triggers
  m_numberAlgorithmTriggers = m_l1GtStablePar->gtNumberPhysTriggers();

  // number of technical triggers
  m_numberTechnicalTriggers = m_l1GtStablePar->gtNumberTechnicalTriggers();

  //    int maxNumberTrigger = std::max(m_numberAlgorithmTriggers,
  //            m_numberTechnicalTriggers);

  //    m_triggerMaskSet.reserve(maxNumberTrigger);
  //    m_prescaleFactorSet.reserve(maxNumberTrigger);

  // get / update the prescale factors from the EventSetup

  m_l1GtPfAlgo = &evSetup.getData(m_l1GtPfAlgoToken);

  m_prescaleFactorsAlgoTrig = &(m_l1GtPfAlgo->gtPrescaleFactors());

  m_l1GtPfTech = &evSetup.getData(m_l1GtPfTechToken);

  m_prescaleFactorsTechTrig = &(m_l1GtPfTech->gtPrescaleFactors());

  // get / update the trigger mask from the EventSetup

  m_l1GtTmAlgo = &evSetup.getData(m_l1GtTmAlgoToken);

  m_triggerMaskAlgoTrig = &(m_l1GtTmAlgo->gtTriggerMask());

  m_l1GtTmTech = &evSetup.getData(m_l1GtTmTechToken);

  m_triggerMaskTechTrig = &(m_l1GtTmTech->gtTriggerMask());

  m_l1GtTmVetoAlgo = &evSetup.getData(m_l1GtTmVetoAlgoToken);

  m_triggerMaskVetoAlgoTrig = &(m_l1GtTmVetoAlgo->gtTriggerMask());

  m_l1GtTmVetoTech = &evSetup.getData(m_l1GtTmVetoTechToken);

  m_triggerMaskVetoTechTrig = &(m_l1GtTmVetoTech->gtTriggerMask());

  // get / update the trigger menu from the EventSetup

  m_l1GtMenu = &evSetup.getData(m_l1GtMenuToken);
  m_algorithmMap = &(m_l1GtMenu->gtAlgorithmMap());
  m_algorithmAliasMap = &(m_l1GtMenu->gtAlgorithmAliasMap());
  m_technicalTriggerMap = &(m_l1GtMenu->gtTechnicalTriggerMap());
}

void L1GtTriggerMenuTester::associateL1SeedsHltPath(const edm::Run& iRun, const edm::EventSetup& evSetup) {
  bool hltChanged = true;

  if (m_hltConfig.init(iRun, evSetup, m_hltProcessName, hltChanged)) {
    // if init returns TRUE, initialization has succeeded!
    if (hltChanged) {
      // HLT configuration has actually changed wrt the previous run
      m_hltTableName = m_hltConfig.tableName();

      edm::LogVerbatim("L1GtTriggerMenuTester") << "\nHLT ConfDB menu name: \n   " << m_hltTableName << std::endl;

      // loop over trigger paths, get the HLTLevel1GTSeed logical expression, and add the path to
      // each L1 trigger

      m_hltPathsForL1AlgorithmTrigger.resize(m_numberAlgorithmTriggers);
      m_hltPathsForL1TechnicalTrigger.resize(m_numberTechnicalTriggers);

      m_algoTriggerSeedNotInL1Menu.reserve(m_numberAlgorithmTriggers);
      m_techTriggerSeedNotInL1Menu.reserve(m_numberTechnicalTriggers);

      for (unsigned int iHlt = 0; iHlt < m_hltConfig.size(); ++iHlt) {
        const std::string& hltPathName = m_hltConfig.triggerName(iHlt);

        const std::vector<std::pair<bool, std::string> >& hltL1Seed = m_hltConfig.hltL1GTSeeds(hltPathName);

        unsigned int numberHltL1GTSeeds = hltL1Seed.size();

        edm::LogVerbatim("L1GtTriggerMenuTester") << "\nPath: " << hltPathName << " :    <== " << numberHltL1GTSeeds
                                                  << " HLTLevel1GTSeed module(s)" << std::endl;

        for (unsigned int iSeedModule = 0; iSeedModule < numberHltL1GTSeeds; ++iSeedModule) {
          // one needs a non-const logical expression... TODO check why
          std::string m_l1SeedsLogicalExpression = (hltL1Seed[iSeedModule]).second;

          edm::LogVerbatim("L1GtTriggerMenuTester") << "      '" << m_l1SeedsLogicalExpression << "'";

          // parse logical expression

          if (m_l1SeedsLogicalExpression != "L1GlobalDecision") {
            // check also the logical expression - add/remove spaces if needed
            L1GtLogicParser m_l1AlgoLogicParser = L1GtLogicParser(m_l1SeedsLogicalExpression);

            // list of required algorithms for seeding
            std::vector<L1GtLogicParser::OperandToken> m_l1AlgoSeeds = m_l1AlgoLogicParser.expressionSeedsOperandList();
            size_t l1AlgoSeedsSize = m_l1AlgoSeeds.size();

            edm::LogVerbatim("L1GtTriggerMenuTester") << " :    <== " << l1AlgoSeedsSize << " L1 seeds" << std::endl;

            // for each algorithm trigger, check if it is in the L1 menu, get the bit number
            // and add path to the vector of strings for that bit number

            for (size_t i = 0; i < l1AlgoSeedsSize; ++i) {
              const std::string& trigNameOrAlias = (m_l1AlgoSeeds[i]).tokenName;

              CItAlgo itAlgo = m_algorithmAliasMap->find(trigNameOrAlias);
              if (itAlgo != m_algorithmAliasMap->end()) {
                int bitNr = (itAlgo->second).algoBitNumber();

                (m_hltPathsForL1AlgorithmTrigger.at(bitNr)).push_back(hltPathName);

                edm::LogVerbatim("L1GtTriggerMenuTester")
                    << "         " << trigNameOrAlias << " bit " << bitNr << std::endl;

              } else {
                if (m_noThrowIncompatibleMenu) {
                  edm::LogVerbatim("L1GtTriggerMenuTester")
                      << "         " << trigNameOrAlias << " trigger not in L1 menu " << m_l1GtMenu->gtTriggerMenuName()
                      << std::endl;

                  m_algoTriggerSeedNotInL1Menu.push_back(trigNameOrAlias);

                } else {
                  throw cms::Exception("FailModule")
                      << "\nAlgorithm  " << trigNameOrAlias
                      << ", requested as seed by a HLT path, not found in the L1 trigger menu\n   "
                      << m_l1GtMenu->gtTriggerMenuName() << "\nIncompatible L1 and HLT menus.\n"
                      << std::endl;
                }
              }
            }
          }
        }
      }
    }
  } else {
    // if init returns FALSE, initialization has NOT succeeded, which indicates a problem
    // with the file and/or code and needs to be investigated!
    edm::LogError("MyAnalyzer") << " HLT config extraction failure with process name " << m_hltProcessName;
  }
}

// printing template for a trigger group
void L1GtTriggerMenuTester::printTriggerGroup(const std::string& trigGroupName,
                                              const std::map<std::string, const L1GtAlgorithm*>& trigGroup,
                                              const bool compactPrint,
                                              const bool printPfsRates) {
  // FIXME get values - either read from a specific L1 menu file, or from
  std::string lumiVal1 = "5.0E33";
  std::string lumiVal2 = "7.0E33";
  std::string trigComment;

  int trigPfVal1 = 0;
  int trigPfVal2 = 0;

  int trigRateVal1 = 0;
  int trigRateVal2 = 0;

  // cumulative list of L1 triggers not used as seed by HLT
  std::vector<std::string> algoTriggerNotSeed;
  algoTriggerNotSeed.reserve(m_numberAlgorithmTriggers);

  std::vector<std::string> techTriggerNotSeed;
  techTriggerNotSeed.reserve(m_numberTechnicalTriggers);

  // force a page break before each group
  edm::LogVerbatim("L1GtTriggerMenuTesterWiki") << "<p style=\"page-break-before: always\">&nbsp;</p>";

  edm::LogVerbatim("L1GtTriggerMenuTesterWiki") << "\n---++++ " << trigGroupName << "\n" << std::endl;

  if (compactPrint) {
    edm::LogVerbatim("L1GtTriggerMenuTesterWiki")
        << "|  *Trigger Name*  |  *Trigger Alias*  |  *Bit*  |  *Comments*  |" << std::endl;

  } else {
    if (printPfsRates) {
      edm::LogVerbatim("L1GtTriggerMenuTesterWiki") << "|  *Trigger Name*  |  *Trigger Alias*  |  *Bit*  |  "
                                                       "*Luminosity*  ||||  *Seed for !HLT path(s)*  |  *Comments*  |"
                                                    << std::endl;

      edm::LogVerbatim("L1GtTriggerMenuTesterWiki")
          << "|^|^|^|  *" << lumiVal1 << "*  ||  *" << lumiVal2 << "*  ||  **  |  **  |" << std::endl;

      edm::LogVerbatim("L1GtTriggerMenuTesterWiki")
          << "|^|^|^|  *PF*  |  *Rate*  |  *PF*  |  *Rate*  |  ** |  **  |" << std::endl;

    } else {
      edm::LogVerbatim("L1GtTriggerMenuTesterWiki")
          << "|  *Trigger Name*  |  *Trigger Alias*  |  *Bit*  |  *Seed for !HLT path(s)*  |" << std::endl;
    }
  }

  for (CItAlgoP itAlgo = trigGroup.begin(); itAlgo != trigGroup.end(); itAlgo++) {
    const std::string& aName = (itAlgo->second)->algoName();
    const std::string& aAlias = (itAlgo->second)->algoAlias();
    const int& bitNumber = (itAlgo->second)->algoBitNumber();

    // concatenate in a string, to simplify the next print instruction
    std::string seedsHlt;
    if (m_useHltMenu) {
      const std::vector<std::string>& hltPaths = m_hltPathsForL1AlgorithmTrigger.at(bitNumber);

      if (hltPaths.empty()) {
        algoTriggerNotSeed.push_back(aAlias);
        seedsHlt = "<font color = \"red\">Not used as seed by any !HLT path</font>";
      } else {
        for (std::vector<std::string>::const_iterator strIter = hltPaths.begin(); strIter != hltPaths.end();
             ++strIter) {
          seedsHlt = seedsHlt + (*strIter) + "<BR>";
        }
      }
    }

    if (compactPrint) {
      edm::LogVerbatim("L1GtTriggerMenuTesterWiki")
          << "|" << std::left << "[[" << (m_htmlFile + "#" + aName) << "][ " << aName << "]] "
          << "  |" << aAlias << "  |  " << bitNumber << "| |" << std::endl;

    } else {
      if (printPfsRates) {
        edm::LogVerbatim("L1GtTriggerMenuTesterWiki")
            << "|" << std::left << "[[" << (m_htmlFile + "#" + aName) << "][ " << aName << "]] "
            << "  |" << aAlias << "  |  " << bitNumber << "|  " << ((trigPfVal1 != 0) ? trigPfVal1 : 0) << "  |  "
            << ((trigRateVal1 != 0) ? trigRateVal1 : 0) << "  |  " << ((trigPfVal2 != 0) ? trigPfVal2 : 0) << "  |  "
            << ((trigRateVal2 != 0) ? trigRateVal2 : 0) << "  |  " << seedsHlt << "  |  " << trigComment << "  |"
            << std::endl;

      } else {
        edm::LogVerbatim("L1GtTriggerMenuTesterWiki")
            << "|" << std::left << "[[" << (m_htmlFile + "#" + aName) << "][ " << aName << "]] "
            << "  |" << aAlias << "  |  " << bitNumber << "|" << seedsHlt << "  |  " << std::endl;
      }
    }
  }

  edm::LogVerbatim("L1GtTriggerMenuTesterWiki")
      << "\n"
      << trigGroupName << ": " << (trigGroup.size()) << " bits defined." << std::endl;

  if (m_useHltMenu && (!compactPrint)) {
    edm::LogVerbatim("L1GtTriggerMenuTesterWiki")
        << "\n Algorithm triggers from " << trigGroupName << " not used as seeds by !HLT:" << std::endl;

    if (!algoTriggerNotSeed.empty()) {
      for (std::vector<std::string>::const_iterator strIter = algoTriggerNotSeed.begin();
           strIter != algoTriggerNotSeed.end();
           ++strIter) {
        edm::LogVerbatim("L1GtTriggerMenuTesterWiki") << "   * " << (*strIter) << std::endl;
      }

    } else {
      edm::LogVerbatim("L1GtTriggerMenuTesterWiki") << "   * none" << std::endl;
    }
  }
}

/// printing in Wiki format
void L1GtTriggerMenuTester::printWiki() {
  //
  // print menu, prescale factors and trigger mask in wiki format
  //

  // L1 GT prescale factors for algorithm triggers

  std::vector<int> prescaleFactorsAlgoTrig = m_prescaleFactorsAlgoTrig->at(m_indexPfSet);

  // L1 GT prescale factors for technical triggers

  std::vector<int> prescaleFactorsTechTrig = m_prescaleFactorsTechTrig->at(m_indexPfSet);

  // use another map <int, L1GtAlgorithm> to get the menu sorted after bit number
  // both algorithm and bit numbers are unique
  typedef std::map<int, const L1GtAlgorithm*>::const_iterator CItBit;

  //    algorithm triggers

  std::map<int, const L1GtAlgorithm*> algoBitToAlgo;

  std::map<std::string, const L1GtAlgorithm*> jetAlgoTrig;
  std::map<std::string, const L1GtAlgorithm*> egammaAlgoTrig;
  std::map<std::string, const L1GtAlgorithm*> esumAlgoTrig;
  std::map<std::string, const L1GtAlgorithm*> muonAlgoTrig;
  std::map<std::string, const L1GtAlgorithm*> crossAlgoTrig;
  std::map<std::string, const L1GtAlgorithm*> bkgdAlgoTrig;

  int algoTrigNumber = 0;
  int freeAlgoTrigNumber = 0;

  int jetAlgoTrigNumber = 0;
  int egammaAlgoTrigNumber = 0;
  int esumAlgoTrigNumber = 0;
  int muonAlgoTrigNumber = 0;
  int crossAlgoTrigNumber = 0;
  int bkgdAlgoTrigNumber = 0;

  for (CItAlgo itAlgo = m_algorithmMap->begin(); itAlgo != m_algorithmMap->end(); itAlgo++) {
    const int bitNumber = (itAlgo->second).algoBitNumber();
    const std::string& algName = (itAlgo->second).algoName();

    algoBitToAlgo[bitNumber] = &(itAlgo->second);

    algoTrigNumber++;

    // per category

    const ConditionMap& conditionMap = (m_l1GtMenu->gtConditionMap()).at((itAlgo->second).algoChipNumber());

    const std::vector<L1GtLogicParser::TokenRPN>& rpnVector = (itAlgo->second).algoRpnVector();
    const L1GtLogicParser::OperationType condOperand = L1GtLogicParser::OP_OPERAND;

    std::list<L1GtObject> listObjects;

    for (size_t i = 0; i < rpnVector.size(); ++i) {
      if ((rpnVector[i]).operation == condOperand) {
        const std::string& cndName = (rpnVector[i]).operand;

        // search the condition in the condition list

        bool foundCond = false;

        CItCond itCond = conditionMap.find(cndName);
        if (itCond != conditionMap.end()) {
          foundCond = true;

          // loop through object types and add them to the list

          const std::vector<L1GtObject>& objType = (itCond->second)->objectType();

          for (std::vector<L1GtObject>::const_iterator itObject = objType.begin(); itObject != objType.end();
               itObject++) {
            listObjects.push_back(*itObject);

            edm::LogVerbatim("L1GtTriggerMenuTester") << (*itObject) << std::endl;
          }

          // FIXME for XML parser, add GtExternal to objType correctly
          if ((itCond->second)->condCategory() == CondExternal) {
            listObjects.push_back(GtExternal);
          }
        }

        if (!foundCond) {
          // it should never be happen, all conditions are in the maps
          throw cms::Exception("FailModule") << "\nCondition " << cndName << " not found in the condition map"
                                             << " for chip number " << ((itAlgo->second).algoChipNumber()) << std::endl;
        }
      }
    }

    // eliminate duplicates
    listObjects.sort();
    listObjects.unique();

    // add the algorithm to the corresponding group

    bool jetGroup = false;
    bool egammaGroup = false;
    bool esumGroup = false;
    bool muonGroup = false;
    bool crossGroup = false;
    bool bkgdGroup = false;

    for (std::list<L1GtObject>::const_iterator itObj = listObjects.begin(); itObj != listObjects.end(); ++itObj) {
      switch (*itObj) {
        case Mu: {
          muonGroup = true;
        }

        break;
        case NoIsoEG: {
          egammaGroup = true;
        }

        break;
        case IsoEG: {
          egammaGroup = true;
        }

        break;
        case CenJet: {
          jetGroup = true;
        }

        break;
        case ForJet: {
          jetGroup = true;
        }

        break;
        case TauJet: {
          jetGroup = true;
        }

        break;
        case ETM: {
          esumGroup = true;

        }

        break;
        case ETT: {
          esumGroup = true;

        }

        break;
        case HTT: {
          esumGroup = true;

        }

        break;
        case HTM: {
          esumGroup = true;

        }

        break;
        case JetCounts: {
          // do nothing - not available
        }

        break;
        case HfBitCounts: {
          bkgdGroup = true;
        }

        break;
        case HfRingEtSums: {
          bkgdGroup = true;
        }

        break;
        case GtExternal: {
          bkgdGroup = true;
        }

        break;
        case TechTrig:
        case Castor:
        case BPTX:
        default: {
          // should not arrive here

          edm::LogVerbatim("L1GtTriggerMenuTester") << "\n      Unknown object of type " << *itObj << std::endl;
        } break;
      }
    }

    int sumGroup = jetGroup + egammaGroup + esumGroup + muonGroup + crossGroup + bkgdGroup;

    if (sumGroup > 1) {
      crossAlgoTrig[algName] = &(itAlgo->second);
    } else {
      if (jetGroup) {
        jetAlgoTrig[algName] = &(itAlgo->second);

      } else if (egammaGroup) {
        egammaAlgoTrig[algName] = &(itAlgo->second);

      } else if (esumGroup && (listObjects.size() > 1)) {
        crossAlgoTrig[algName] = &(itAlgo->second);

      } else if (esumGroup) {
        esumAlgoTrig[algName] = &(itAlgo->second);

      } else if (muonGroup) {
        muonAlgoTrig[algName] = &(itAlgo->second);

      } else if (bkgdGroup) {
        bkgdAlgoTrig[algName] = &(itAlgo->second);

      } else {
        // do nothing
      }
    }

    edm::LogVerbatim("L1GtTriggerMenuTester")
        << algName << " sum: " << sumGroup << " size: " << listObjects.size() << std::endl;
  }

  freeAlgoTrigNumber = m_numberAlgorithmTriggers - algoTrigNumber;

  jetAlgoTrigNumber = jetAlgoTrig.size();
  egammaAlgoTrigNumber = egammaAlgoTrig.size();
  esumAlgoTrigNumber = esumAlgoTrig.size();
  muonAlgoTrigNumber = muonAlgoTrig.size();
  crossAlgoTrigNumber = crossAlgoTrig.size();
  bkgdAlgoTrigNumber = bkgdAlgoTrig.size();

  //    technical triggers
  std::map<int, const L1GtAlgorithm*> techBitToAlgo;

  int techTrigNumber = 0;
  int freeTechTrigNumber = 0;

  for (CItAlgo itAlgo = m_technicalTriggerMap->begin(); itAlgo != m_technicalTriggerMap->end(); itAlgo++) {
    int bitNumber = (itAlgo->second).algoBitNumber();
    techBitToAlgo[bitNumber] = &(itAlgo->second);

    techTrigNumber++;
  }

  freeTechTrigNumber = m_numberTechnicalTriggers - techTrigNumber;

  // name of the attached HTML file
  if (!m_overwriteHtmlFile) {
    std::string menuName = m_l1GtMenu->gtTriggerMenuImplementation();

    // replace "/" with "_"
    std::replace(menuName.begin(), menuName.end(), '/', '_');
    m_htmlFile = "%ATTACHURL%/" + menuName + ".html";
  } else {
    m_htmlFile = "%ATTACHURL%/" + m_htmlFile;
  }

  // header for printing algorithms

  edm::LogVerbatim("L1GtTriggerMenuTesterWiki")
      << "\n   ********** L1 Trigger Menu - printing in wiki format  ********** \n\n"
      << "\n---+++ L1 menu identification\n"
      << "\n|L1 Trigger Menu Interface: |!" << m_l1GtMenu->gtTriggerMenuInterface() << " |"
      << "\n|L1 Trigger Menu Name: |!" << m_l1GtMenu->gtTriggerMenuName() << " |"
      << "\n|L1 Trigger Menu Implementation: |!" << m_l1GtMenu->gtTriggerMenuImplementation() << " |"
      << "\n|Associated L1 scale DB key: |!" << m_l1GtMenu->gtScaleDbKey() << " |"
      << "\n\n"
      << std::flush << std::endl;

  // Overview page
  edm::LogVerbatim("L1GtTriggerMenuTesterWiki") << "\n---+++ Summary\n" << std::endl;
  edm::LogVerbatim("L1GtTriggerMenuTesterWiki")
      << "   * Number of algorithm triggers: " << algoTrigNumber << " defined, 128 possible." << std::endl;
  edm::LogVerbatim("L1GtTriggerMenuTesterWiki")
      << "   * Number of technical triggers: " << techTrigNumber << " defined,  64 possible.<BR><BR>" << std::endl;

  edm::LogVerbatim("L1GtTriggerMenuTesterWiki")
      << "   * Number of free bits for algorithm triggers: " << freeAlgoTrigNumber << std::endl;
  edm::LogVerbatim("L1GtTriggerMenuTesterWiki")
      << "   * Number of free bits for technical triggers: " << freeTechTrigNumber << "<BR>" << std::endl;

  edm::LogVerbatim("L1GtTriggerMenuTesterWiki") << "\nNumber of algorithm triggers per trigger group\n" << std::endl;
  edm::LogVerbatim("L1GtTriggerMenuTesterWiki") << "    | *Trigger group* | *Number of bits used*|" << std::endl;
  edm::LogVerbatim("L1GtTriggerMenuTesterWiki")
      << "    | Jet algorithm triggers: |  " << jetAlgoTrigNumber << "|" << std::endl;
  edm::LogVerbatim("L1GtTriggerMenuTesterWiki")
      << "    | EGamma algorithm triggers: |  " << egammaAlgoTrigNumber << "|" << std::endl;
  edm::LogVerbatim("L1GtTriggerMenuTesterWiki")
      << "    | Energy sum algorithm triggers: |  " << esumAlgoTrigNumber << "|" << std::endl;
  edm::LogVerbatim("L1GtTriggerMenuTesterWiki")
      << "    | Muon algorithm triggers: |  " << muonAlgoTrigNumber << "|" << std::endl;
  edm::LogVerbatim("L1GtTriggerMenuTesterWiki")
      << "    | Cross algorithm triggers: |  " << crossAlgoTrigNumber << "|" << std::endl;
  edm::LogVerbatim("L1GtTriggerMenuTesterWiki")
      << "    | Background algorithm triggers: |  " << bkgdAlgoTrigNumber << "|" << std::endl;

  // force a page break
  edm::LogVerbatim("L1GtTriggerMenuTesterWiki") << "<p style=\"page-break-before: always\">&nbsp;</p>";

  // compact print - without HLT path
  bool compactPrint = true;

  edm::LogVerbatim("L1GtTriggerMenuTesterWiki") << "\n---+++ List of algorithm triggers sorted by trigger groups\n"
                                                << std::endl;

  // Jet algorithm triggers
  printTriggerGroup("Jet algorithm triggers", jetAlgoTrig, compactPrint, m_printPfsRates);

  // EGamma algorithm triggers
  printTriggerGroup("EGamma algorithm triggers", egammaAlgoTrig, compactPrint, m_printPfsRates);

  // Energy sum algorithm triggers
  printTriggerGroup("Energy sum algorithm triggers", esumAlgoTrig, compactPrint, m_printPfsRates);

  // Muon algorithm triggers
  printTriggerGroup("Muon algorithm triggers", muonAlgoTrig, compactPrint, m_printPfsRates);

  // Cross algorithm triggers
  printTriggerGroup("Cross algorithm triggers", crossAlgoTrig, compactPrint, m_printPfsRates);

  // Background algorithm triggers
  printTriggerGroup("Background algorithm triggers", bkgdAlgoTrig, compactPrint, m_printPfsRates);

  // force a page break
  edm::LogVerbatim("L1GtTriggerMenuTesterWiki") << "<p style=\"page-break-before: always\">&nbsp;</p>";

  edm::LogVerbatim("L1GtTriggerMenuTesterWiki") << "\n---+++ List of algorithm triggers sorted by bits\n" << std::endl;

  edm::LogVerbatim("L1GtTriggerMenuTesterWiki") << "| *Algorithm* | *Alias* | *Bit number* |" << std::endl;

  for (CItBit itBit = algoBitToAlgo.begin(); itBit != algoBitToAlgo.end(); itBit++) {
    int bitNumber = itBit->first;
    std::string aName = (itBit->second)->algoName();
    std::string aAlias = (itBit->second)->algoAlias();

    edm::LogVerbatim("L1GtTriggerMenuTesterWiki")
        << "|" << std::left << "[[" << (m_htmlFile + "#" + aName) << "][ " << aName << "]] "
        << "  |" << aAlias << "  |  " << bitNumber << "| |" << std::endl;
  }

  // force a page break
  edm::LogVerbatim("L1GtTriggerMenuTesterWiki") << "<p style=\"page-break-before: always\">&nbsp;</p>";
  edm::LogVerbatim("L1GtTriggerMenuTesterWiki") << "\n---+++ List of technical triggers\n" << std::endl;

  edm::LogVerbatim("L1GtTriggerMenuTesterWiki") << "| *Technical trigger* | *Bit number* |" << std::endl;

  for (CItBit itBit = techBitToAlgo.begin(); itBit != techBitToAlgo.end(); itBit++) {
    int bitNumber = itBit->first;
    std::string aName = (itBit->second)->algoName();
    std::string aAlias = (itBit->second)->algoAlias();

    edm::LogVerbatim("L1GtTriggerMenuTesterWiki")
        << "|!" << std::left << aName << "  |  " << std::right << bitNumber << " |" << std::endl;
  }

  // force a page break
  edm::LogVerbatim("L1GtTriggerMenuTesterWiki") << "<p style=\"page-break-before: always\">&nbsp;</p>";

  // compact print false: with HLT path association, if the parameter m_useHltMenu is true
  // otherwise, we have no association computed

  if (m_useHltMenu) {
    compactPrint = false;
  } else {
    return;
  }

  edm::LogVerbatim("L1GtTriggerMenuTesterWiki")
      << "\n---+++ List of algorithm triggers sorted by trigger groups, including !HLT path association \n"
      << std::endl;

  edm::LogVerbatim("L1GtTriggerMenuTesterWiki")
      << "\n The following !HLT menu was used to associate the !HLT path to the L1 algorithm triggers:\n    "
      << std::endl;
  edm::LogVerbatim("L1GtTriggerMenuTesterWiki") << m_hltTableName << std::endl;

  // Jet algorithm triggers
  printTriggerGroup("Jet algorithm triggers", jetAlgoTrig, compactPrint, m_printPfsRates);

  // EGamma algorithm triggers
  printTriggerGroup("EGamma algorithm triggers", egammaAlgoTrig, compactPrint, m_printPfsRates);

  // Energy sum algorithm triggers
  printTriggerGroup("Energy sum algorithm triggers", esumAlgoTrig, compactPrint, m_printPfsRates);

  // Muon algorithm triggers
  printTriggerGroup("Muon algorithm triggers", muonAlgoTrig, compactPrint, m_printPfsRates);

  // Cross algorithm triggers
  printTriggerGroup("Cross algorithm triggers", crossAlgoTrig, compactPrint, m_printPfsRates);

  // Background algorithm triggers
  printTriggerGroup("Background algorithm triggers", bkgdAlgoTrig, compactPrint, m_printPfsRates);
}

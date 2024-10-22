/** \class HLTPrescaleProvider
 *
 * See header file for documentation
 *
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTPrescaleProvider.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <cassert>
#include <sstream>

static const bool useL1EventSetup(true);
static const bool useL1GtTriggerMenuLite(false);
static const unsigned char countMax(2);

bool HLTPrescaleProvider::init(const edm::Run& iRun,
                               const edm::EventSetup& iSetup,
                               const std::string& processName,
                               bool& changed) {
  inited_ = true;

  count_[0] = 0;
  count_[1] = 0;
  count_[2] = 0;
  count_[3] = 0;
  count_[4] = 0;

  const bool result(hltConfigProvider_.init(iRun, iSetup, processName, changed));

  const unsigned int l1tType(hltConfigProvider_.l1tType());
  if (l1tType == 1) {
    checkL1GtUtils();
    /// L1 GTA V3: https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideL1TriggerL1GtUtils#Version_3
    l1GtUtils_->getL1GtRunCache(iRun, iSetup, useL1EventSetup, useL1GtTriggerMenuLite);
  } else if (l1tType == 2) {
    checkL1TGlobalUtil();
    l1tGlobalUtil_->retrieveL1Setup(iSetup);
  } else {
    edm::LogError("HLTPrescaleProvider") << " Unknown L1T Type " << l1tType << " - prescales will not be avaiable!";
  }

  return result;
}

L1GtUtils const& HLTPrescaleProvider::l1GtUtils() const {
  checkL1GtUtils();
  return *l1GtUtils_;
}

l1t::L1TGlobalUtil const& HLTPrescaleProvider::l1tGlobalUtil() const {
  checkL1TGlobalUtil();
  return *l1tGlobalUtil_;
}

int HLTPrescaleProvider::prescaleSet(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if (!inited_) {
    throw cms::Exception("LogicError") << "HLTPrescaleProvider::prescaleSet,\n"
                                          "HLTPrescaleProvider::init was not called at beginRun\n";
  }
  const unsigned int l1tType(hltConfigProvider_.l1tType());
  if (l1tType == 1) {
    checkL1GtUtils();

    // return hltPrescaleTable_.set();
    l1GtUtils_->getL1GtRunCache(iEvent, iSetup, useL1EventSetup, useL1GtTriggerMenuLite);
    int errorTech(0);
    const int psfsiTech(l1GtUtils_->prescaleFactorSetIndex(iEvent, L1GtUtils::TechnicalTrigger, errorTech));
    int errorPhys(0);
    const int psfsiPhys(l1GtUtils_->prescaleFactorSetIndex(iEvent, L1GtUtils::AlgorithmTrigger, errorPhys));
    assert(psfsiTech == psfsiPhys);
    if ((errorTech == 0) && (errorPhys == 0) && (psfsiTech >= 0) && (psfsiPhys >= 0) && (psfsiTech == psfsiPhys)) {
      return psfsiPhys;
    } else {
      /// error - notify user!
      if (count_[0] < countMax) {
        count_[0] += 1;
        edm::LogError("HLTPrescaleProvider")
            << " Using processName '" << hltConfigProvider_.processName() << "':"
            << " Error in determining HLT prescale set index from L1 data using L1GtUtils:"
            << " Tech/Phys error = " << errorTech << "/" << errorPhys << " Tech/Phys psfsi = " << psfsiTech << "/"
            << psfsiPhys;
      }
      return -1;
    }
  } else if (l1tType == 2) {
    checkL1TGlobalUtil();
    l1tGlobalUtil_->retrieveL1Event(iEvent, iSetup);
    return static_cast<int>(l1tGlobalUtil_->prescaleColumn());
  } else {
    if (count_[0] < countMax) {
      count_[0] += 1;
      edm::LogError("HLTPrescaleProvider")
          << " Using processName '" << hltConfigProvider_.processName() << "':"
          << " Unknown L1T Type " << l1tType << " - can not determine prescale set index!";
    }
    return -1;
  }
}

template <>
FractionalPrescale HLTPrescaleProvider::convertL1PS(double val) const {
  int numer = static_cast<int>(val * kL1PrescaleDenominator_ + 0.5);
  static constexpr double kL1RoundingEpsilon = 0.001;
  if (std::abs(numer - val * kL1PrescaleDenominator_) > kL1RoundingEpsilon) {
    edm::LogWarning("ValueError") << " Error, L1 prescale val " << val
                                  << " does not appear to precisely expressable as int / " << kL1PrescaleDenominator_
                                  << ", using a FractionalPrescale is a loss of precision";
  }

  return {numer, kL1PrescaleDenominator_};
}

double HLTPrescaleProvider::getL1PrescaleValue(const edm::Event& iEvent,
                                               const edm::EventSetup& iSetup,
                                               const std::string& trigger) {
  // get L1T prescale - works only for those hlt trigger paths with
  // exactly one L1GT seed module which has exactly one L1T name as seed

  double result = -1;

  const unsigned int l1tType(hltConfigProvider_.l1tType());
  if (l1tType == 1) {
    checkL1GtUtils();
    const unsigned int nL1GTSeedModules(hltConfigProvider_.hltL1GTSeeds(trigger).size());
    if (nL1GTSeedModules == 0) {
      // no L1 seed module on path hence no L1 seed hence formally no L1 prescale
      result = 1;
    } else if (nL1GTSeedModules == 1) {
      l1GtUtils_->getL1GtRunCache(iEvent, iSetup, useL1EventSetup, useL1GtTriggerMenuLite);
      const std::string l1tname(hltConfigProvider_.hltL1GTSeeds(trigger).at(0).second);
      if (l1tname == l1tGlobalDecisionKeyword_) {
        LogDebug("HLTPrescaleProvider") << "Undefined L1T prescale for HLT path: '" << trigger << "' with L1T seed '"
                                        << l1tGlobalDecisionKeyword_ << "' (keyword for global decision of L1T)";
        result = -1;
      } else {
        int l1error(0);
        result = l1GtUtils_->prescaleFactor(iEvent, l1tname, l1error);
        if (l1error != 0) {
          if (count_[1] < countMax) {
            count_[1] += 1;
            edm::LogError("HLTPrescaleProvider")
                << " Error in determining L1T prescale for HLT path: '" << trigger << "' with L1T seed: '" << l1tname
                << "' using L1GtUtils: error code = " << l1error << "." << std::endl
                << " Note: for this method ('prescaleValues'), only a single L1T name (and not a bit number)"
                << " is allowed as seed!" << std::endl
                << " For seeds being complex logical expressions, try the new method 'prescaleValuesInDetail'."
                << std::endl;
          }
          result = -1;
        }
      }
    } else {
      /// error - can't handle properly multiple L1GTSeed modules
      if (count_[2] < countMax) {
        count_[2] += 1;
        std::string dump("'" + hltConfigProvider_.hltL1GTSeeds(trigger).at(0).second + "'");
        for (unsigned int i = 1; i != nL1GTSeedModules; ++i) {
          dump += " * '" + hltConfigProvider_.hltL1GTSeeds(trigger).at(i).second + "'";
        }
        edm::LogError("HLTPrescaleProvider")
            << " Error in determining L1T prescale for HLT path: '" << trigger << "' has multiple L1GTSeed modules, "
            << nL1GTSeedModules << ", with L1 seeds: " << dump
            << ". (Note: at most one L1GTSeed module is allowed for a proper determination of the L1T prescale!)";
      }
      result = -1;
    }
  } else if (l1tType == 2) {
    checkL1TGlobalUtil();
    const unsigned int nL1TSeedModules(hltConfigProvider_.hltL1TSeeds(trigger).size());
    if (nL1TSeedModules == 0) {
      // no L1 seed module on path hence no L1 seed hence formally no L1 prescale
      result = 1;
    } else if (nL1TSeedModules == 1) {
      //    l1tGlobalUtil_->retrieveL1Event(iEvent,iSetup);
      const std::string l1tname(hltConfigProvider_.hltL1TSeeds(trigger).at(0));
      if (l1tname == l1tGlobalDecisionKeyword_) {
        LogDebug("HLTPrescaleProvider") << "Undefined L1T prescale for HLT path: '" << trigger << "' with L1T seed '"
                                        << l1tGlobalDecisionKeyword_ << "' (keyword for global decision of L1T)";
        result = -1;
      } else {
        bool l1error(!l1tGlobalUtil_->getPrescaleByName(l1tname, result));
        if (l1error) {
          if (count_[1] < countMax) {
            count_[1] += 1;
            edm::LogError("HLTPrescaleProvider")
                << " Error in determining L1T prescale for HLT path: '" << trigger << "' with L1T seed: '" << l1tname
                << "' using L1TGlobalUtil: error cond = " << l1error << "." << std::endl
                << " Note: for this method ('prescaleValues'), only a single L1T name (and not a bit number)"
                << " is allowed as seed!" << std::endl
                << " For seeds being complex logical expressions, try the new method 'prescaleValuesInDetail'."
                << std::endl;
          }
          result = -1;
        }
      }
    } else {
      /// error - can't handle properly multiple L1TSeed modules
      if (count_[2] < countMax) {
        count_[2] += 1;
        std::string dump("'" + hltConfigProvider_.hltL1TSeeds(trigger).at(0) + "'");
        for (unsigned int i = 1; i != nL1TSeedModules; ++i) {
          dump += " * '" + hltConfigProvider_.hltL1TSeeds(trigger).at(i) + "'";
        }
        edm::LogError("HLTPrescaleProvider")
            << " Error in determining L1T prescale for HLT path: '" << trigger << "' has multiple L1TSeed modules, "
            << nL1TSeedModules << ", with L1T seeds: " << dump
            << ". (Note: at most one L1TSeed module is allowed for a proper determination of the L1T prescale!)";
      }
      result = -1;
    }
  } else {
    if (count_[1] < countMax) {
      count_[1] += 1;
      edm::LogError("HLTPrescaleProvider") << " Unknown L1T Type " << l1tType << " - can not determine L1T prescale! ";
    }
    result = -1;
  }

  return result;
}

std::vector<std::pair<std::string, double>> HLTPrescaleProvider::getL1PrescaleValueInDetail(
    const edm::Event& iEvent, const edm::EventSetup& iSetup, const std::string& trigger) {
  std::vector<std::pair<std::string, double>> result;

  const unsigned int l1tType(hltConfigProvider_.l1tType());
  if (l1tType == 1) {
    checkL1GtUtils();

    const unsigned int nL1GTSeedModules(hltConfigProvider_.hltL1GTSeeds(trigger).size());
    if (nL1GTSeedModules == 0) {
      // no L1 seed module on path hence no L1 seed hence formally no L1 prescale
      result.clear();
    } else if (nL1GTSeedModules == 1) {
      l1GtUtils_->getL1GtRunCache(iEvent, iSetup, useL1EventSetup, useL1GtTriggerMenuLite);
      const std::string l1tname(hltConfigProvider_.hltL1GTSeeds(trigger).at(0).second);
      if (l1tname == l1tGlobalDecisionKeyword_) {
        LogDebug("HLTPrescaleProvider") << "Undefined L1T prescales for HLT path: '" << trigger << "' with L1T seed '"
                                        << l1tGlobalDecisionKeyword_ << "' (keyword for global decision of L1T)";
        result.clear();
      } else {
        L1GtUtils::LogicalExpressionL1Results l1Logical(l1tname, *l1GtUtils_);
        l1Logical.logicalExpressionRunUpdate(iEvent.getRun(), iSetup, l1tname);
        const std::vector<std::pair<std::string, int>>& errorCodes(l1Logical.errorCodes(iEvent));
        auto resultInt = l1Logical.prescaleFactors();
        result.clear();
        result.reserve(resultInt.size());
        for (const auto& entry : resultInt) {
          result.push_back(entry);
        }
        int l1error(l1Logical.isValid() ? 0 : 1);
        for (auto const& errorCode : errorCodes) {
          l1error += std::abs(errorCode.second);
        }
        if (l1error != 0) {
          if (count_[3] < countMax) {
            count_[3] += 1;
            std::ostringstream message;
            message << " Error in determining L1T prescales for HLT path: '" << trigger << "' with complex L1T seed: '"
                    << l1tname << "' using L1GtUtils: " << std::endl
                    << " isValid=" << l1Logical.isValid() << " l1tname/error/prescale " << errorCodes.size()
                    << std::endl;
            for (unsigned int i = 0; i < errorCodes.size(); ++i) {
              message << " " << i << ":" << errorCodes[i].first << "/" << errorCodes[i].second << "/"
                      << result[i].second;
            }
            message << ".";
            edm::LogError("HLTPrescaleProvider") << message.str();
          }
          result.clear();
        }
      }
    } else {
      /// error - can't handle properly multiple L1GTSeed modules
      if (count_[4] < countMax) {
        count_[4] += 1;
        std::string dump("'" + hltConfigProvider_.hltL1GTSeeds(trigger).at(0).second + "'");
        for (unsigned int i = 1; i != nL1GTSeedModules; ++i) {
          dump += " * '" + hltConfigProvider_.hltL1GTSeeds(trigger).at(i).second + "'";
        }
        edm::LogError("HLTPrescaleProvider")
            << " Error in determining L1T prescale for HLT path: '" << trigger << "' has multiple L1GTSeed modules, "
            << nL1GTSeedModules << ", with L1 seeds: " << dump
            << ". (Note: at most one L1GTSeed module is allowed for a proper determination of the L1T prescale!)";
      }
      result.clear();
    }
  } else if (l1tType == 2) {
    checkL1TGlobalUtil();
    const unsigned int nL1TSeedModules(hltConfigProvider_.hltL1TSeeds(trigger).size());
    if (nL1TSeedModules == 0) {
      // no L1 seed module on path hence no L1 seed hence formally no L1 prescale
      result.clear();
    } else if (nL1TSeedModules == 1) {
      std::string l1tname(hltConfigProvider_.hltL1TSeeds(trigger).at(0));
      if (l1tname == l1tGlobalDecisionKeyword_) {
        LogDebug("HLTPrescaleProvider") << "Undefined L1T prescales for HLT path: '" << trigger << "' with L1T seed '"
                                        << l1tGlobalDecisionKeyword_ << "' (keyword for global decision of L1T)";
        result.clear();
      } else {
        GlobalLogicParser l1tGlobalLogicParser = GlobalLogicParser(l1tname);
        const std::vector<GlobalLogicParser::OperandToken> l1tSeeds = l1tGlobalLogicParser.expressionSeedsOperandList();
        int l1error(0);
        double l1tPrescale(-1);
        result.clear();
        result.reserve(l1tSeeds.size());
        for (auto const& i : l1tSeeds) {
          const string& l1tSeed = i.tokenName;
          if (!l1tGlobalUtil_->getPrescaleByName(l1tSeed, l1tPrescale)) {
            l1error += 1;
          }
          result.push_back(std::pair<std::string, double>(l1tSeed, l1tPrescale));
        }
        if (l1error != 0) {
          if (count_[3] < countMax) {
            count_[3] += 1;
            string l1name = l1tname;
            std::ostringstream message;
            message << " Error in determining L1T prescales for HLT path: '" << trigger << "' with complex L1T seed: '"
                    << l1tname << "' using L1TGlobalUtil: " << std::endl
                    << " isValid=" << l1tGlobalLogicParser.checkLogicalExpression(l1name) << " l1tname/error/prescale "
                    << l1tSeeds.size() << std::endl;
            for (unsigned int i = 0; i < l1tSeeds.size(); ++i) {
              const string& l1tSeed = l1tSeeds[i].tokenName;
              message << " " << i << ":" << l1tSeed << "/" << l1tGlobalUtil_->getPrescaleByName(l1tSeed, l1tPrescale)
                      << "/" << result[i].second;
            }
            message << ".";
            edm::LogError("HLTPrescaleProvider") << message.str();
          }
          result.clear();
        }
      }
    } else {
      /// error - can't handle properly multiple L1TSeed modules
      if (count_[4] < countMax) {
        count_[4] += 1;
        std::string dump("'" + hltConfigProvider_.hltL1TSeeds(trigger).at(0) + "'");
        for (unsigned int i = 1; i != nL1TSeedModules; ++i) {
          dump += " * '" + hltConfigProvider_.hltL1TSeeds(trigger).at(i) + "'";
        }
        edm::LogError("HLTPrescaleProvider")
            << " Error in determining L1T prescale for HLT path: '" << trigger << "' has multiple L1TSeed modules, "
            << nL1TSeedModules << ", with L1T seeds: " << dump
            << ". (Note: at most one L1TSeed module is allowed for a proper determination of the L1T prescale!)";
      }
      result.clear();
    }
  } else {
    if (count_[3] < countMax) {
      count_[3] += 1;
      edm::LogError("HLTPrescaleProvider") << " Unknown L1T Type " << l1tType << " - can not determine L1T prescale! ";
    }
    result.clear();
  }

  return result;
}

bool HLTPrescaleProvider::rejectedByHLTPrescaler(const edm::TriggerResults& triggerResults, unsigned int i) const {
  return hltConfigProvider_.moduleType(hltConfigProvider_.moduleLabel(i, triggerResults.index(i))) == "HLTPrescaler";
}

void HLTPrescaleProvider::checkL1GtUtils() const {
  if (!l1GtUtils_) {
    throw cms::Exception("Configuration") << "HLTPrescaleProvider::checkL1GtUtils(),\n"
                                             "Attempt to use L1GtUtils object when none was constructed.\n"
                                             "Possibly the proper era is not configured or\n"
                                             "the module configuration does not use the era properly\n"
                                             "or input is from mixed eras";
  }
}

void HLTPrescaleProvider::checkL1TGlobalUtil() const {
  if (!l1tGlobalUtil_) {
    throw cms::Exception("Configuration") << "HLTPrescaleProvider:::checkL1TGlobalUtil(),\n"
                                             "Attempt to use L1TGlobalUtil object when none was constructed.\n"
                                             "Possibly the proper era is not configured or\n"
                                             "the module configuration does not use the era properly\n"
                                             "or input is from mixed eras";
  }
}

void HLTPrescaleProvider::fillPSetDescription(edm::ParameterSetDescription& desc,
                                              unsigned int stageL1Trigger,
                                              edm::InputTag const& l1tAlgBlkInputTag,
                                              edm::InputTag const& l1tExtBlkInputTag,
                                              bool readPrescalesFromFile) {
  desc.add<unsigned int>("stageL1Trigger", stageL1Trigger);
  L1GtUtils::fillDescription(desc);
  l1t::L1TGlobalUtil::fillDescription(desc, l1tAlgBlkInputTag, l1tExtBlkInputTag, readPrescalesFromFile);
}

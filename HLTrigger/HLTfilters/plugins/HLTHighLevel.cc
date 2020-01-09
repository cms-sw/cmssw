/** \class HLTHighLevel
 *
 * See header file for documentation
 *
 *
 *  \author Martin Grunewald
 *
 */

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/RegexMatch.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/PathContext.h"
#include "FWCore/ServiceRegistry/interface/PlaceInPathContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"

// needed for trigger bits from EventSetup as in ALCARECO paths
#include "CondFormats/HLTObjects/interface/AlCaRecoTriggerBits.h"
#include "CondFormats/DataRecord/interface/AlCaRecoTriggerBitsRcd.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "HLTHighLevel.h"

//
// constructors and destructor
//
HLTHighLevel::HLTHighLevel(const edm::ParameterSet& iConfig)
    : inputTag_(iConfig.getParameter<edm::InputTag>("TriggerResultsTag")),
      inputToken_(consumes<edm::TriggerResults>(inputTag_)),
      triggerNamesID_(),
      andOr_(iConfig.getParameter<bool>("andOr")),
      throw_(iConfig.getParameter<bool>("throw")),
      eventSetupPathsKey_(iConfig.getParameter<std::string>("eventSetupPathsKey")),
      HLTPatterns_(iConfig.getParameter<std::vector<std::string> >("HLTPaths")),
      HLTPathsByName_(),
      HLTPathsByIndex_() {
  // names and slot numbers are computed during the event loop,
  // as they need to access the TriggerNames object via the TriggerResults

  if (!eventSetupPathsKey_.empty()) {
    // If paths come from eventsetup, we must watch for IOV changes.
    if (!HLTPatterns_.empty()) {
      // We do not want double trigger path setting, so throw!
      throw cms::Exception("Configuration")
          << " HLTHighLevel instance: " << iConfig.getParameter<std::string>("@module_label") << "\n configured with "
          << HLTPatterns_.size() << " HLTPaths and\n"
          << " eventSetupPathsKey " << eventSetupPathsKey_ << ", choose either of them.";
    }
    alcaRecotriggerBitsToken_ = esConsumes<AlCaRecoTriggerBits, AlCaRecoTriggerBitsRcd>();
    watchAlCaRecoTriggerBitsRcd_.emplace();
  }
}

//
// member functions
//

void HLTHighLevel::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("TriggerResultsTag", edm::InputTag("TriggerResults", "", "HLT"));
  std::vector<std::string> hltPaths(0);
  // # provide list of HLT paths (or patterns) you want
  desc.add<std::vector<std::string> >("HLTPaths", hltPaths);
  // # not empty => use read paths from AlCaRecoTriggerBitsRcd via this key
  desc.add<std::string>("eventSetupPathsKey", "");
  // # how to deal with multiple triggers: True (OR) accept if ANY is true, False (AND) accept if ALL are true
  desc.add<bool>("andOr", true);
  // # throw exception on unknown path names
  desc.add<bool>("throw", true);
  descriptions.add("hltHighLevel", desc);
}

// Initialize the internal trigger path representation (names and indices) from the
// patterns specified in the configuration.
// This needs to be called once at startup, whenever the trigger table has changed
// or in case of paths from eventsetup and IOV changed
void HLTHighLevel::init(const edm::TriggerResults& result,
                        const edm::Event& event,
                        const edm::EventSetup& iSetup,
                        const edm::TriggerNames& triggerNames) {
  unsigned int n;

  // clean up old data
  HLTPathsByName_.clear();
  HLTPathsByIndex_.clear();

  // Overwrite paths from EventSetup via AlCaRecoTriggerBitsRcd if configured:
  if (!eventSetupPathsKey_.empty()) {
    HLTPatterns_ = this->pathsFromSetup(eventSetupPathsKey_, event, iSetup);
  }

  if (HLTPatterns_.empty()) {
    // for empty input vector, default to all HLT trigger paths
    n = result.size();
    HLTPathsByName_.resize(n);
    HLTPathsByIndex_.resize(n);
    for (unsigned int i = 0; i < n; ++i) {
      HLTPathsByName_[i] = triggerNames.triggerName(i);
      HLTPathsByIndex_[i] = i;
    }
  } else {
    // otherwise, expand wildcards in trigger names...
    for (auto const& pattern : HLTPatterns_) {
      if (edm::is_glob(pattern)) {
        // found a glob pattern, expand it
        std::vector<std::vector<std::string>::const_iterator> matches =
            edm::regexMatch(triggerNames.triggerNames(), pattern);
        if (matches.empty()) {
          // pattern does not match any trigger paths
          if (throw_)
            throw cms::Exception("Configuration")
                << "requested pattern \"" << pattern << "\" does not match any HLT paths";
          else
            edm::LogInfo("Configuration") << "requested pattern \"" << pattern << "\" does not match any HLT paths";
        } else {
          // store the matching patterns
          for (auto const& match : matches)
            HLTPathsByName_.push_back(*match);
        }
      } else {
        // found a trigger name, just copy it
        HLTPathsByName_.push_back(pattern);
      }
    }
    n = HLTPathsByName_.size();

    // ...and get hold of trigger indices
    bool valid = false;
    HLTPathsByIndex_.resize(n);
    for (unsigned int i = 0; i < HLTPathsByName_.size(); i++) {
      HLTPathsByIndex_[i] = triggerNames.triggerIndex(HLTPathsByName_[i]);
      if (HLTPathsByIndex_[i] < result.size()) {
        valid = true;
      } else {
        // trigger path not found
        HLTPathsByIndex_[i] = (unsigned int)-1;
        if (throw_)
          throw cms::Exception("Configuration") << "requested HLT path \"" << HLTPathsByName_[i] << "\" does not exist";
        else
          edm::LogInfo("Configuration") << "requested HLT path \"" << HLTPathsByName_[i] << "\" does not exist";
      }
    }

    if (not valid) {
      // no point in throwing - if requested, it should have already happened
      edm::LogWarning("Configuration")
          << "none of the requested paths and pattern match any HLT path - no events will be selected";
    }
  }

  // report on what is finally used
  LogDebug("HLTHighLevel") << "HLT trigger paths: " + inputTag_.encode() << " - Number of paths: " << n
                           << " - andOr mode: " << andOr_ << " - throw mode: " << throw_;

  LogTrace("HLTHighLevel") << "The HLT trigger paths (# index name):";
  for (unsigned int i = 0; i < n; ++i)
    if (HLTPathsByIndex_[i] == (unsigned int)-1)
      LogTrace("HLTHighLevel") << "    n/a   " << HLTPathsByName_[i];
    else
      LogTrace("HLTHighLevel") << "    " << std::setw(4) << HLTPathsByIndex_[i] << "  " << HLTPathsByName_[i];
}

// ------------ getting paths from EventSetup  ------------
std::vector<std::string> HLTHighLevel::pathsFromSetup(const std::string& key,
                                                      const edm::Event& event,
                                                      const edm::EventSetup& iSetup) const {
  // Get map of strings to concatenated list of names of HLT paths from EventSetup:
  const auto& triggerBits = iSetup.getData(alcaRecotriggerBitsToken_);
  typedef std::map<std::string, std::string> TriggerMap;
  const TriggerMap& triggerMap = triggerBits.m_alcarecoToTrig;

  auto listIter = triggerMap.find(key);
  if (listIter == triggerMap.end()) {
    throw cms::Exception("Configuration")
        << " HLTHighLevel [instance: " << moduleLabel() << " - path: " << pathName(event)
        << "]: No triggerList with key " << key << " in AlCaRecoTriggerBitsRcd";
  }

  // We must avoid a map<string,vector<string> > in DB for performance reason,
  // so the paths are mapped into one string that we have to decompose:
  return triggerBits.decompose(listIter->second);
}

// ------------ method called to produce the data  ------------
bool HLTHighLevel::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace std;
  using namespace edm;

  // get hold of TriggerResults Object
  Handle<TriggerResults> trh;
  iEvent.getByToken(inputToken_, trh);
  if (trh.isValid()) {
    LogDebug("HLTHighLevel") << "TriggerResults found, number of HLT paths: " << trh->size();
  } else {
    LogError("HLTHighLevel") << "TriggerResults product " << inputTag_.encode()
                             << " not found - returning result=false!";
    return false;
  }

  // init the TriggerNames with the TriggerResults
  const edm::TriggerNames& triggerNames = iEvent.triggerNames(*trh);
  bool config_changed = false;
  if (triggerNamesID_ != triggerNames.parameterSetID()) {
    triggerNamesID_ = triggerNames.parameterSetID();
    config_changed = true;
  }

  // (re)run the initialization stuff if
  // - this is the first event
  // - or the HLT table has changed
  // - or selected trigger bits come from AlCaRecoTriggerBitsRcd and these changed
  if (config_changed or (watchAlCaRecoTriggerBitsRcd_ and watchAlCaRecoTriggerBitsRcd_->check(iSetup))) {
    this->init(*trh, iEvent, iSetup, triggerNames);
  }
  unsigned int n = HLTPathsByName_.size();
  unsigned int nbad = 0;
  unsigned int fired = 0;

  // count invalid and fired triggers
  for (unsigned int i = 0; i < n; i++)
    if (HLTPathsByIndex_[i] == (unsigned int)-1)
      ++nbad;
    else if (trh->accept(HLTPathsByIndex_[i]))
      ++fired;

  if ((nbad > 0) and (config_changed or throw_)) {
    // only generate the error message if it's actually going to be used
    std::string message;

    for (unsigned int i = 0; i < n; i++)
      if (HLTPathsByIndex_[i] == (unsigned int)-1)
        message += HLTPathsByName_[i] + " ";

    if (config_changed) {
      LogTrace("HLTHighLevel") << " HLTHighLevel [instance: " << moduleLabel() << " - path: " << pathName(iEvent)
                               << "] configured with " << nbad << "/" << n << " unknown HLT path names: " << message;
    }

    if (throw_) {
      throw cms::Exception("Configuration")
          << " HLTHighLevel [instance: " << moduleLabel() << " - path: " << pathName(iEvent) << "] configured with "
          << nbad << "/" << n << " unknown HLT path names: " << message;
    }
  }

  // Boolean filter result (always at least one trigger)
  const bool accept((fired > 0) and (andOr_ or (fired == n - nbad)));
  LogDebug("HLTHighLevel") << "Accept = " << std::boolalpha << accept;

  return accept;
}

std::string const& HLTHighLevel::pathName(const edm::Event& event) const {
  return event.moduleCallingContext()->placeInPathContext()->pathContext()->pathName();
}

std::string const& HLTHighLevel::moduleLabel() const { return moduleDescription().moduleLabel(); }

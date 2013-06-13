#include "DQMOffline/Trigger/interface/HLTTauDQMPathPlotter2.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include <boost/regex.hpp>

#include<cstdio>
#include<sstream>
#include<algorithm>

namespace {
  // Used as a helper only in this file
  class HLTPath {
  public:
    HLTPath(const std::string& name, const boost::smatch& what):
      name_(name)
    {
      char buffer[10];
      for(int i=0; i<10; ++i) {
        snprintf(buffer, 10, "tr%d", i);
        boost::ssub_match sm = what[buffer];
        if(sm.length() == 0)
          break;

        try {
          thresholds_.push_back(std::stoi(sm.str())); // C++11
        } catch(std::invalid_argument& e) {
          throw cms::Exception("Configuration") << "Interpreting regex of path " << name << ", threshold " << buffer << ": unable to convert '" << sm.str() << "' to integer";
        } catch(std::out_of_range& e) {
          throw cms::Exception("Configuration") << "Interpreting regex of path " << name << ", threshold " << buffer << ": '" << sm.str() << "' is out of int range";
        }
      }
    }

    bool isBetterThan(const HLTPath& other, const HLTConfigProvider& HLTCP)  const {
      // First compare prescales
      // Search for prescale set where either is enabled
      // If other is disabled (prescale = 0), pick the enabled one
      // If prescales are different, pick the one with smaller
      // If prescales are same, continue to compare thresholds
      for(unsigned int iSet = 0; iSet < HLTCP.prescaleSize(); ++iSet) {
        unsigned int prescale = HLTCP.prescaleValue(iSet, name_);
        unsigned int prescaleOther = HLTCP.prescaleValue(iSet, other.name_);
        if(prescale == 0 && prescaleOther == 0)
          continue;
        if(prescale == 0)
          return false;
        if(prescaleOther == 0)
          return true;

        if(prescale != prescaleOther)
          return prescale < prescaleOther;
        break;
      }

      // Then thresholds
      if(thresholds_.size() != other.thresholds_.size())
        throw cms::Exception("Configuration") << "Comparing path " << name_ << " and " << other.name_ << ", they have different numbers of thresholds (" << thresholds_.size() << " != " << other.thresholds_.size() << ")";
      for(size_t i=0; i<thresholds_.size(); ++i) {
        if(thresholds_[i] != other.thresholds_[i])
          return thresholds_[i] < other.thresholds_[i];
      }

      // Nothing left, what to do? In principle this is an error
      throw cms::Exception("Configuration") << "Comparing path " << name_ << " and " << other.name_ << ", unable to tell which to choose. Improve your input regex!";
      return true;
    }

    typedef HLTTauDQMPathPlotter2::FilterIndex FilterIndex;

    std::vector<FilterIndex> interestingFilters(const HLTConfigProvider& HLTCP, bool doRefAnalysis, const std::vector<boost::regex>& ignoreFilterTypes, const std::vector<boost::regex>& ignoreFilterNames) const {
      const std::vector<std::string>& moduleLabels = HLTCP.moduleLabels(name_);
      std::vector<FilterIndex> selectedFilters;
      std::vector<std::string> leptonTauFilters;

      // Ignore all "Selector"s, for ref-analysis keep only those with saveTags=True
      // Also record HLT2(Electron|Muon)(PF)?Tau module names
      for(std::vector<std::string>::const_iterator iLabel = moduleLabels.begin(); iLabel != moduleLabels.end(); ++iLabel) {
        if(HLTCP.moduleEDMType(*iLabel) != "EDFilter")
          continue;
        const std::string type = HLTCP.moduleType(*iLabel);
        if(type.find("Selector") != std::string::npos)
          continue;
        if(type == "HLTTriggerTypeFilter" || type == "HLTBool")
          continue;
        if(doRefAnalysis && !HLTCP.saveTags(*iLabel))
          continue;
        if(type == "HLT2ElectronPFTau" || type == "HLT2MuonPFTau" || type == "HLT2ElectronTau" || type == "HLT2MuonTau")
          leptonTauFilters.emplace_back(*iLabel);
        else if(type.find("Electron") != std::string::npos || type.find("Egamma") != std::string::npos || type.find("Muon") != std::string::npos)
          continue;
        selectedFilters.emplace_back(*iLabel, iLabel-moduleLabels.begin());
      }

      // Insert the last filters of lepton legs
      for(const std::string& leptonTauLabel: leptonTauFilters) {
        const edm::ParameterSet& pset = HLTCP.modulePSet(leptonTauLabel);
        std::string input1 = pset.getParameter<edm::InputTag>("inputTag1").label();
        std::string input2 = pset.getParameter<edm::InputTag>("inputTag2").label();
        unsigned idx1 = HLTCP.moduleIndex(name_, input1);
        unsigned idx2 = HLTCP.moduleIndex(name_, input2);

        auto func = [&](const FilterIndex& a, unsigned idxb) {
          return std::get<1>(a) < idxb;
        };
        std::vector<FilterIndex>::iterator found = std::lower_bound(selectedFilters.begin(), selectedFilters.end(), idx1, func);
        if(found == selectedFilters.end() || std::get<1>(*found) != idx1)
          selectedFilters.emplace(found, input1, idx1);
        found = std::lower_bound(selectedFilters.begin(), selectedFilters.end(), idx2, func);
        if(found == selectedFilters.end() || std::get<1>(*found) != idx2)
          selectedFilters.emplace(found, input2, idx2);
      }

      // Remove filters ignored by their type
      std::vector<FilterIndex>::iterator selectedFiltersEnd = std::remove_if(selectedFilters.begin(), selectedFilters.end(), [&](const FilterIndex& labelIndex) {
          for(const boost::regex& re: ignoreFilterTypes) {
            if(boost::regex_search(HLTCP.moduleType(std::get<0>(labelIndex)), re))
              return true;
          }
          return false;
        });
      // Remove filters ignored by their label
      selectedFiltersEnd = std::remove_if(selectedFilters.begin(), selectedFiltersEnd, [&](const FilterIndex& labelIndex) {
          for(const boost::regex& re: ignoreFilterNames) {
            if(boost::regex_search(std::get<0>(labelIndex), re))
              return true;
          }
          return false;
        });


      std::vector<FilterIndex> ret;
      ret.reserve(selectedFiltersEnd-selectedFilters.begin());
      std::move(selectedFilters.begin(), selectedFiltersEnd, std::back_inserter(ret));
      return ret;
    }

    const std::string& name() const { return name_; }

  private:
    std::string name_;

    std::vector<int> thresholds_;
  };


}


HLTTauDQMPathPlotter2::HLTTauDQMPathPlotter2(const edm::ParameterSet& pset, bool doRefAnalysis, const std::string& dqmBaseFolder, const HLTConfigProvider& HLTCP):
  doRefAnalysis_(doRefAnalysis)
{
  dqmBaseFolder_ = dqmBaseFolder;

  // Parse configuration
  std::vector<std::string> regexs;
  std::vector<boost::regex> ignoreFilterTypes;
  std::vector<boost::regex> ignoreFilterNames;
  try {
    triggerTag_         = pset.getUntrackedParameter<std::string>("DQMFolder");
    regexs              = pset.getUntrackedParameter<std::vector<std::string> >("Path");

    std::vector<std::string> ignoreFilterTypesTmp;
    std::vector<std::string> ignoreFilterNamesTmp;
    ignoreFilterTypesTmp   = pset.getUntrackedParameter<std::vector<std::string> >("IgnoreFilterTypes");
    ignoreFilterNamesTmp   = pset.getUntrackedParameter<std::vector<std::string> >("IgnoreFilterNames");
    ignoreFilterTypes.reserve(ignoreFilterTypesTmp.size());
    ignoreFilterNames.reserve(ignoreFilterNamesTmp.size());
    for(const std::string& str: ignoreFilterTypesTmp)
      ignoreFilterTypes.emplace_back(str);
    for(const std::string& str: ignoreFilterNamesTmp)
      ignoreFilterNames.emplace_back(str);

  } catch(cms::Exception& e) {
    edm::LogInfo("HLTTauDQMOffline") << "HLTTauDQMPathPlotter2::HLTTauDQMPathPlotter2(): " << e.what();
    validity_ = false;
    return;
  }

  // Identify the correct HLT path
  if(!HLTCP.inited()) {
    edm::LogInfo("HLTTauDQMOffline") << "HLTTauDQMPathPlotter2::HLTTauDQMPathPlotter2(): HLTConfigProvider is not initialized!";
    validity_ = false;
    return;
  }

  // Search path candidates
  std::vector<HLTPath> foundPaths;
  try {
    const std::vector<std::string>& triggerNames = HLTCP.triggerNames();
    for(const std::string& regexStr: regexs) {
      //std::cout << regexStr << std::endl;
      const boost::regex re(regexStr);
      boost::smatch what;

      for(const std::string& path: triggerNames) {
        if(boost::regex_match(path, what, re)) {
          foundPaths.emplace_back(path, what);
        }
      }
      if(!foundPaths.empty())
        break;
    }
  } catch(cms::Exception& e) {
    edm::LogInfo("HLTTauDQMOffline") << "HLTTauDQMPathPlotter2::HLTTauDQMPathPlotter2(): " << e.what();
    validity_ = false;
    return;
  }
  if(foundPaths.empty()) {
    std::stringstream ss;
    for(std::vector<std::string>::const_iterator iRegex = regexs.begin(); iRegex != regexs.end(); ++iRegex) {
      if(iRegex != regexs.begin())
        ss << ",";
      ss << *iRegex;
    }
    edm::LogInfo("HLTTauDQMOffline") << "HLTTauDQMPathPlotter2::HLTTauDQMPathPlotter2(): did not find any paths matching to regexes " << ss.str();
    validity_ = false;
    return;
  }

  // If more than one, find the best match
  std::vector<HLTPath>::const_iterator thePath = foundPaths.begin();
  std::vector<HLTPath>::const_iterator iPath = thePath;
  ++iPath;
  for(; iPath != foundPaths.end(); ++iPath) {
    if(!thePath->isBetterThan(*iPath, HLTCP))
      thePath = iPath;
  }
  std::cout << "Chose path " << thePath->name() << std::endl;

  // Get the filters
  filterIndices_ = thePath->interestingFilters(HLTCP, doRefAnalysis, ignoreFilterTypes, ignoreFilterNames);
  std::cout << "  Filters" << std::endl;
  for(const FilterIndex& nameIndex: filterIndices_)
    std::cout << "    " << std::get<1>(nameIndex) << " " << std::get<0>(nameIndex) << "  " << HLTCP.moduleType(std::get<0>(nameIndex)) << std::endl;

  // Set path index
  pathIndex_ = HLTCP.triggerIndex(thePath->name());

  // Book histograms
  if(store_) {
    store_->setCurrentFolder(triggerTag());
    store_->removeContents();

    hAcceptedEvents_ = store_->book1D("EventsPerFilter","Accepted Events per Filter;;entries", filterIndices_.size(), 0, filterIndices_.size());
    for(size_t i=0; i<filterIndices_.size(); ++i) {
      hAcceptedEvents_->setBinLabel(i+1, std::get<0>(filterIndices_[i]));
    }
  }

  validity_ = true;
}


HLTTauDQMPathPlotter2::~HLTTauDQMPathPlotter2() {}

void HLTTauDQMPathPlotter2::analyze(const edm::TriggerResults& triggerResults, const trigger::TriggerEvent& triggerEvent, const std::map<int, LVColl>& refCollection) {


  // Events per filter
  size_t fillUntilBin = 0;
  if(triggerResults.accept(pathIndex_)) {
    fillUntilBin = filterIndices_.size()+1;
    //std::cout << "Event passed" << std::endl;
  }
  else {
    unsigned int firstFailedFilter = triggerResults.index(pathIndex_);
    for(size_t i=0; i<filterIndices_.size(); ++i) {
      if(std::get<1>(filterIndices_[i]) < firstFailedFilter) {
        fillUntilBin = i+1;
      }
      else {
        //std::cout << "Decision-making filter " << firstFailedFilter << " this " << std::get<1>(filterIndices_[i]) << std::endl;
        break;
      }
    }
  }
  for(size_t bin=0; bin<fillUntilBin; ++bin) {
    hAcceptedEvents_->Fill(bin+0.5);
  }

}

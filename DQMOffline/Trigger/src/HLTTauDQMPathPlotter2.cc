#include "DQMOffline/Trigger/interface/HLTTauDQMPathPlotter2.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include <boost/regex.hpp>

#include<cstdio>
#include<sstream>

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

    const std::string& name() const { return name_; }

  private:
    std::string name_;

    std::vector<int> thresholds_;
  };
}


HLTTauDQMPathPlotter2::HLTTauDQMPathPlotter2(const edm::ParameterSet& pset, bool doRefAnalysis, const std::string& dqmBaseFolder, const HLTConfigProvider& HLTCP):
  doRefAnalysis_(doRefAnalysis)
{

  // Parse configuration
  std::string dqmFolder;
  std::vector<std::string> regexs;
  try {
    dqmFolder           = pset.getUntrackedParameter<std::string>("DQMFolder");
    regexs              = pset.getUntrackedParameter<std::vector<std::string> >("Path");
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
    for(std::vector<std::string>::const_iterator iRegex = regexs.begin(); iRegex != regexs.end(); ++iRegex) {
      //std::cout << *iRegex << std::endl;
      const boost::regex re(*iRegex);
      boost::smatch what;

      for(std::vector<std::string>::const_iterator iPath = triggerNames.begin(); iPath != triggerNames.end(); ++iPath) {
        if(boost::regex_match(*iPath, what, re)) {
          foundPaths.push_back(HLTPath(*iPath, what));
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
  edm::LogInfo("HLTTauDQMOffline") << "HLTTauDQMPathPlotter2::HLTTauDQMPathPlotter2(): chose path " << thePath->name();

  validity_ = true;
}


HLTTauDQMPathPlotter2::~HLTTauDQMPathPlotter2() {}

void HLTTauDQMPathPlotter2::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, const std::map<int, LVColl>& refCollection) {
}

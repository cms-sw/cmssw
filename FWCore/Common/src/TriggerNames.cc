
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <algorithm>

namespace {
  struct PairSort {
    bool operator()(std::pair<std::string_view, unsigned int> const& iLHS,
                    std::pair<std::string_view, unsigned int> const& iRHS) const {
      return iLHS.first < iRHS.first;
    }
    bool operator()(std::string_view iLHS, std::pair<std::string_view, unsigned int> const& iRHS) const {
      return iLHS < iRHS.first;
    }
    bool operator()(std::pair<std::string_view, unsigned int> const& iLHS, std::string_view iRHS) const {
      return iLHS.first < iRHS;
    }
  };
}  // namespace
namespace edm {

  TriggerNames::TriggerNames(edm::ParameterSet const& pset)
      : psetID_{pset.id()}, triggerNames_{pset.getParameter<Strings>("@trigger_paths")} {
    initializeTriggerIndex();
  }

  TriggerNames::TriggerNames(TriggerNames const& iOther)
      : psetID_{iOther.psetID_}, triggerNames_{iOther.triggerNames_} {
    initializeTriggerIndex();
  }

  TriggerNames& TriggerNames::operator=(TriggerNames const& iOther) {
    TriggerNames temp(iOther);
    *this = std::move(temp);
    return *this;
  }

  void TriggerNames::initializeTriggerIndex() {
    unsigned int index = 0;
    indexMap_.reserve(triggerNames_.size());
    for (auto const& name : triggerNames_) {
      indexMap_.emplace_back(name, index);
      ++index;
    }
    std::sort(indexMap_.begin(), indexMap_.end(), PairSort());
  }

  TriggerNames::Strings const& TriggerNames::triggerNames() const { return triggerNames_; }

  std::string const& TriggerNames::triggerName(unsigned int index) const { return triggerNames_.at(index); }

  unsigned int TriggerNames::triggerIndex(std::string_view name) const {
    auto found = std::equal_range(indexMap_.begin(), indexMap_.end(), name, PairSort());
    if (found.first == found.second)
      return indexMap_.size();
    return found.first->second;
  }

  std::size_t TriggerNames::size() const { return triggerNames_.size(); }

  ParameterSetID const& TriggerNames::parameterSetID() const { return psetID_; }
}  // namespace edm

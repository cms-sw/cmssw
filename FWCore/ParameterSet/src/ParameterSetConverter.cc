/*----------------------------------------------------------------------

ParameterSetConverter.cc

----------------------------------------------------------------------*/
//------------------------------------------------------------
// Adapts parameter sets by value (fileFormatVersion_.value() < 12) to parameter sets by reference
// Adapts untracked @trigger_paths (fileFormatVersion_.value() < 13) to tracked @trigger_paths.

#include "FWCore/ParameterSet/interface/ParameterSetConverter.h"
#include <iterator>
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ParameterSet/src/split.h"
#include "FWCore/Utilities/interface/Algorithms.h"

namespace edm {

  namespace {
    void insertIntoReplace(ParameterSetConverter::StringMap& replace,
                           std::string const& fromPrefix,
                           std::string const& from,
                           std::string const& fromPostfix,
                           std::string const& toPrefix,
                           std::string const& to,
                           std::string const& toPostfix) {
      replace.insert(std::make_pair(fromPrefix + from + fromPostfix, toPrefix + to + toPostfix));
    }
  }  // namespace

  MainParameterSet::MainParameterSet(ParameterSetID const& oldID, std::string const& psetString)
      : oldID_(oldID),
        parameterSet_(psetString),
        paths_(parameterSet_.getParameter<StringVector>("@paths")),
        endPaths_(),
        triggerPaths_() {
    if (parameterSet_.existsAs<StringVector>("@end_paths")) {
      endPaths_ = (parameterSet_.getParameter<StringVector>("@end_paths"));
    }
    for (StringVector::const_iterator i = paths_.begin(), e = paths_.end(); i != e; ++i) {
      if (!search_all(endPaths_, *i)) {
        triggerPaths_.insert(*i);
      }
    }
  }

  MainParameterSet::~MainParameterSet() {}

  TriggerPath::TriggerPath(ParameterSet const& pset)
      : parameterSet_(pset), tPaths_(parameterSet_.getParameter<StringVector>("@trigger_paths")), triggerPaths_() {
    for (StringVector::const_iterator i = tPaths_.begin(), e = tPaths_.end(); i != e; ++i) {
      triggerPaths_.insert(*i);
    }
  }

  TriggerPath::~TriggerPath() {}

  //------------------------------------------------------------

  ParameterSetConverter::ParameterSetConverter(ParameterSetMap const& psetMap,
                                               ParameterSetIdConverter& idConverter,
                                               bool alreadyByReference)
      : parameterSets_(), mainParameterSets_(), triggerPaths_(), replace_(), parameterSetIdConverter_(idConverter) {
    for (ParameterSetMap::const_iterator i = psetMap.begin(), iEnd = psetMap.end(); i != iEnd; ++i) {
      parameterSets_.push_back(std::make_pair(i->second.pset(), i->first));
    }
    if (alreadyByReference) {
      noConvertParameterSets();
    } else {
      replace_.insert(std::make_pair(std::string("=+p({})"), std::string("=+q({})")));
      convertParameterSets();
    }
    for (std::vector<MainParameterSet>::iterator j = mainParameterSets_.begin(), jEnd = mainParameterSets_.end();
         j != jEnd;
         ++j) {
      for (std::vector<TriggerPath>::iterator i = triggerPaths_.begin(), iEnd = triggerPaths_.end(); i != iEnd; ++i) {
        if (i->triggerPaths_ == j->triggerPaths_) {
          j->parameterSet_.addParameter("@trigger_paths", i->parameterSet_);
          break;
        }
      }
    }
    for (std::vector<MainParameterSet>::iterator i = mainParameterSets_.begin(), iEnd = mainParameterSets_.end();
         i != iEnd;
         ++i) {
      ParameterSet& pset = i->parameterSet_;
      pset.registerIt();
      ParameterSetID newID(pset.id());
      if (i->oldID_ != newID && i->oldID_ != ParameterSetID()) {
        parameterSetIdConverter_.insert(std::make_pair(i->oldID_, newID));
      }
    }
  }

  ParameterSetConverter::~ParameterSetConverter() {}

  void ParameterSetConverter::noConvertParameterSets() {
    for (StringWithIDList::iterator i = parameterSets_.begin(), iEnd = parameterSets_.end(); i != iEnd; ++i) {
      if (i->first.find("@all_sources") != std::string::npos) {
        mainParameterSets_.push_back(MainParameterSet(i->second, i->first));
      } else {
        ParameterSet pset(i->first);
        pset.setID(i->second);
        pset::Registry::instance()->insertMapped(pset);
        if (i->first.find("@trigger_paths") != std::string::npos) {
          triggerPaths_.push_back(pset);
        }
      }
    }
  }

  void ParameterSetConverter::convertParameterSets() {
    std::string const comma(",");
    std::string const rparam(")");
    std::string const rvparam("})");
    std::string const loldparam("=+P(");
    std::string const loldvparam("=+p({");
    std::string const lparam("=+Q(");
    std::string const lvparam("=+q({");
    bool doItAgain = false;
    for (StringMap::const_iterator j = replace_.begin(), jEnd = replace_.end(); j != jEnd; ++j) {
      for (StringWithIDList::iterator i = parameterSets_.begin(), iEnd = parameterSets_.end(); i != iEnd; ++i) {
        for (std::string::size_type it = i->first.find(j->first); it != std::string::npos;
             it = i->first.find(j->first)) {
          i->first.replace(it, j->first.size(), j->second);
          doItAgain = true;
        }
      }
    }
    for (StringWithIDList::iterator i = parameterSets_.begin(), iEnd = parameterSets_.end(); i != iEnd; ++i) {
      if (i->first.find("+P") == std::string::npos && i->first.find("+p") == std::string::npos) {
        if (i->first.find("@all_sources") != std::string::npos) {
          mainParameterSets_.push_back(MainParameterSet(i->second, i->first));
        } else {
          ParameterSet pset(i->first);
          pset.registerIt();
          std::string& from = i->first;
          std::string to;
          ParameterSetID newID(pset.id());
          newID.toString(to);
          insertIntoReplace(replace_, loldparam, from, rparam, lparam, to, rparam);
          insertIntoReplace(replace_, comma, from, comma, comma, to, comma);
          insertIntoReplace(replace_, comma, from, rvparam, comma, to, rvparam);
          insertIntoReplace(replace_, loldvparam, from, comma, lvparam, to, comma);
          insertIntoReplace(replace_, loldvparam, from, rvparam, lvparam, to, rvparam);
          if (i->second != newID && i->second != ParameterSetID()) {
            parameterSetIdConverter_.insert(std::make_pair(i->second, newID));
          }
          if (i->first.find("@trigger_paths") != std::string::npos) {
            triggerPaths_.push_back(pset);
          }
        }
        StringWithIDList::iterator icopy = i;
        ++i;
        parameterSets_.erase(icopy);
        --i;
        doItAgain = true;
      }
    }
    if (!doItAgain && !parameterSets_.empty()) {
      for (auto const& k : parameterSets_) {
        std::vector<std::string_view> pieces;
        split(std::back_inserter(pieces), k.first, '<', ';', '>');
        for (auto const& i : pieces) {
          std::string_view removeName = i.substr(i.find('+'));
          if (removeName.size() >= 4) {
            if (removeName[1] == 'P') {
              std::string psetString(removeName.begin() + 3, removeName.end() - 1);
              parameterSets_.emplace_back(std::move(psetString), ParameterSetID());
              doItAgain = true;
            } else if (removeName[1] == 'p') {
              std::string_view pvec = removeName.substr(3, removeName.size() - 4);
              std::vector<std::string_view> temp;
              split(std::back_inserter(temp), pvec, '{', ',', '}');
              for (auto const& j : temp) {
                parameterSets_.emplace_back(j, ParameterSetID());
              }
              doItAgain = true;
            }
          }
        }
      }
    }
    if (doItAgain) {
      convertParameterSets();
    }
  }
}  // namespace edm

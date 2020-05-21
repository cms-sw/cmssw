/*----------------------------------------------------------------------

ParameterSetConverter.cc

----------------------------------------------------------------------*/
//------------------------------------------------------------
// Adapts parameter sets by value (fileFormatVersion_.value() < 12) to parameter sets by reference
// Adapts untracked @trigger_paths (fileFormatVersion_.value() < 13) to tracked @trigger_paths.

#include "FWCore/ParameterSet/interface/ParameterSetConverter.h"
#include <iterator>
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ParameterSet/interface/split.h"
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
    for (const auto& path : paths_) {
      if (!search_all(endPaths_, path)) {
        triggerPaths_.insert(path);
      }
    }
  }

  MainParameterSet::~MainParameterSet() {}

  TriggerPath::TriggerPath(ParameterSet const& pset)
      : parameterSet_(pset), tPaths_(parameterSet_.getParameter<StringVector>("@trigger_paths")), triggerPaths_() {
    for (const auto& tPath : tPaths_) {
      triggerPaths_.insert(tPath);
    }
  }

  TriggerPath::~TriggerPath() {}

  //------------------------------------------------------------

  ParameterSetConverter::ParameterSetConverter(ParameterSetMap const& psetMap,
                                               ParameterSetIdConverter& idConverter,
                                               bool alreadyByReference)
      : parameterSets_(), mainParameterSets_(), triggerPaths_(), replace_(), parameterSetIdConverter_(idConverter) {
    for (const auto& i : psetMap) {
      parameterSets_.push_back(std::make_pair(i.second.pset(), i.first));
    }
    if (alreadyByReference) {
      noConvertParameterSets();
    } else {
      replace_.insert(std::make_pair(std::string("=+p({})"), std::string("=+q({})")));
      convertParameterSets();
    }
    for (auto& mainParameterSet : mainParameterSets_) {
      for (auto& triggerPath : triggerPaths_) {
        if (triggerPath.triggerPaths_ == mainParameterSet.triggerPaths_) {
          mainParameterSet.parameterSet_.addParameter("@trigger_paths", triggerPath.parameterSet_);
          break;
        }
      }
    }
    for (auto& mainParameterSet : mainParameterSets_) {
      ParameterSet& pset = mainParameterSet.parameterSet_;
      pset.registerIt();
      ParameterSetID newID(pset.id());
      if (mainParameterSet.oldID_ != newID && mainParameterSet.oldID_ != ParameterSetID()) {
        parameterSetIdConverter_.insert(std::make_pair(mainParameterSet.oldID_, newID));
      }
    }
  }

  ParameterSetConverter::~ParameterSetConverter() {}

  void ParameterSetConverter::noConvertParameterSets() {
    for (auto& parameterSet : parameterSets_) {
      if (parameterSet.first.find("@all_sources") != std::string::npos) {
        mainParameterSets_.push_back(MainParameterSet(parameterSet.second, parameterSet.first));
      } else {
        ParameterSet pset(parameterSet.first);
        pset.setID(parameterSet.second);
        pset::Registry::instance()->insertMapped(pset);
        if (parameterSet.first.find("@trigger_paths") != std::string::npos) {
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
    for (const auto& j : replace_) {
      for (auto& parameterSet : parameterSets_) {
        for (std::string::size_type it = parameterSet.first.find(j.first); it != std::string::npos;
             it = parameterSet.first.find(j.first)) {
          parameterSet.first.replace(it, j.first.size(), j.second);
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
        std::list<std::string> pieces;
        split(std::back_inserter(pieces), k.first, '<', ';', '>');
        for (auto& piece : pieces) {
          std::string removeName = piece.substr(piece.find('+'));
          if (removeName.size() >= 4) {
            if (removeName[1] == 'P') {
              std::string psetString(removeName.begin() + 3, removeName.end() - 1);
              parameterSets_.push_back(std::make_pair(psetString, ParameterSetID()));
              doItAgain = true;
            } else if (removeName[1] == 'p') {
              std::string pvec = std::string(removeName.begin() + 3, removeName.end() - 1);
              StringList temp;
              split(std::back_inserter(temp), pvec, '{', ',', '}');
              for (const auto& j : temp) {
                parameterSets_.push_back(std::make_pair(j, ParameterSetID()));
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

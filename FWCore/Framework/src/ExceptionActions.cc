
#include "FWCore/Framework/interface/ExceptionActions.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Utilities/interface/Algorithms.h"

#include <vector>
#include <iostream>

namespace edm {
  namespace exception_actions {
    char const* actionName(ActionCodes code) {
      static constexpr std::array<char const*, LastCode> tab = []() constexpr {
        std::array<char const*, LastCode> table{};
        table[IgnoreCompletely] = "IgnoreCompletely";
        table[Rethrow] = "Rethrow";
        table[TryToContinue] = "TryToContinue";
        return table;
      }();
      return static_cast<unsigned int>(code) < tab.size() ? tab[code] : "UnknownAction";
    }
  }  // namespace exception_actions

  ExceptionToActionTable::ExceptionToActionTable() : map_() { addDefaults(); }

  namespace {
    inline void install(exception_actions::ActionCodes code,
                        ExceptionToActionTable::ActionMap& out,
                        ParameterSet const& pset) {
      typedef std::vector<std::string> vstring;

      // we cannot have parameters in the main process section so look
      // for an untracked (optional) ParameterSet called "options" for
      // now.  Notice that all exceptions (most actally) throw
      // edm::Exception with the configuration category.  This
      // category should probably be more specific or a derived
      // exception type should be used so the catch can be more
      // specific.

      //	cerr << pset.toString() << std::endl;

      ParameterSet const& opts = pset.getUntrackedParameterSet("options");
      //cerr << "looking for " << actionName(code) << std::endl;
      for (auto const& v : opts.getUntrackedParameter<std::vector<std::string>>(actionName(code))) {
        out[v] = code;
      }
    }
  }  // namespace

  ExceptionToActionTable::ExceptionToActionTable(ParameterSet const& pset) : map_() {
    addDefaults();

    install(exception_actions::TryToContinue, map_, pset);
    install(exception_actions::Rethrow, map_, pset);
    install(exception_actions::IgnoreCompletely, map_, pset);
  }

  void ExceptionToActionTable::addDefaults() {
    // populate defaults that are not 'Rethrow'
    // (There are none as of CMSSW_3_4_X.)
    // 'Rethrow' is the default default.
    if (2 <= debugit()) {
      ActionMap::const_iterator ib(map_.begin()), ie(map_.end());
      for (; ib != ie; ++ib) {
        std::cerr << ib->first << ',' << ib->second << '\n';
      }
      std::cerr << std::endl;
    }
  }

  ExceptionToActionTable::~ExceptionToActionTable() {}

  void ExceptionToActionTable::add(std::string const& category, exception_actions::ActionCodes code) {
    map_[category] = code;
  }

  exception_actions::ActionCodes ExceptionToActionTable::find(std::string const& category) const {
    ActionMap::const_iterator i(map_.find(category));
    return i != map_.end() ? i->second : exception_actions::Rethrow;
  }

}  // namespace edm


#include "FWCore/Framework/interface/ExceptionActions.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "boost/lambda/lambda.hpp"

#include <vector>
#include <iostream>

namespace edm {
  namespace exception_actions {
    namespace {
      struct ActionNames {
	ActionNames():table_(LastCode + 1) {
	  table_[IgnoreCompletely] = "IgnoreCompletely";
	  table_[Rethrow] = "Rethrow";
	  table_[SkipEvent] = "SkipEvent";
	  table_[FailPath] = "FailPath";
	}

	typedef std::vector<char const*> Table;
	Table table_;
      };      
    }

    char const* actionName(ActionCodes code) {
      static ActionNames tab;
      return static_cast<unsigned int>(code) < tab.table_.size() ?  tab.table_[code] : "UnknownAction";
    }
  }

  ExceptionToActionTable::ExceptionToActionTable() : map_() {
    addDefaults();
  }

  namespace {
    inline void install(exception_actions::ActionCodes code,
			ExceptionToActionTable::ActionMap& out,
			ParameterSet const& pset) {
      using boost::lambda::_1;
      using boost::lambda::var;
      typedef std::vector<std::string> vstring;

      // we cannot have parameters in the main process section so look
      // for an untracked (optional) ParameterSet called "options" for
      // now.  Notice that all exceptions (most actally) throw
      // edm::Exception with the configuration category.  This
      // category should probably be more specific or a derived
      // exception type should be used so the catch can be more
      // specific.

//	cerr << pset.toString() << std::endl;

      ParameterSet defopts;
      ParameterSet const& opts = pset.getUntrackedParameterSet("options", defopts);
      //cerr << "looking for " << actionName(code) << std::endl;
      vstring v = opts.getUntrackedParameter(actionName(code),vstring());
      for_all(v, var(out)[_1] = code);      

    }  
  }

  ExceptionToActionTable::ExceptionToActionTable(ParameterSet const& pset) : map_() {
    addDefaults();

    install(exception_actions::SkipEvent, map_, pset);
    install(exception_actions::Rethrow, map_, pset);
    install(exception_actions::IgnoreCompletely, map_, pset);
    install(exception_actions::FailPath, map_, pset);
  }

  void ExceptionToActionTable::addDefaults() {
    using namespace boost::lambda;
    // populate defaults that are not 'Rethrow'
    // (There are none as of CMSSW_3_4_X.)
    // 'Rethrow' is the default default.
    if(2 <= debugit()) {
	ActionMap::const_iterator ib(map_.begin()),ie(map_.end());
	for(;ib != ie; ++ib) {
	  std::cerr << ib->first << ',' << ib->second << '\n';
	}
	std::cerr << std::endl;
    }
  }

  ExceptionToActionTable::~ExceptionToActionTable() {
  }

  void ExceptionToActionTable::add(std::string const& category, exception_actions::ActionCodes code) {
    map_[category] = code;
  }

  exception_actions::ActionCodes ExceptionToActionTable::find(std::string const& category) const {
    ActionMap::const_iterator i(map_.find(category));
    return i != map_.end() ? i->second : exception_actions::Rethrow;
  }

}

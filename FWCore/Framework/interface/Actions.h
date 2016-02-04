#ifndef Framework_Actions_h
#define Framework_Actions_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>
#include <map>

namespace edm {
  namespace actions {
    enum ActionCodes {
	IgnoreCompletely=0,
	Rethrow,
	SkipEvent,
	FailPath,
	LastCode
    };

    const char* actionName(ActionCodes code);
  }

  class ActionTable {
  public:
    typedef std::map<std::string, actions::ActionCodes> ActionMap;

    ActionTable();
    explicit ActionTable(const ParameterSet&);
    ~ActionTable();

    void add(const std::string& category, actions::ActionCodes code);
    actions::ActionCodes find(const std::string& category) const;

  private:
    void addDefaults();
    ActionMap map_;
  };
}
#endif

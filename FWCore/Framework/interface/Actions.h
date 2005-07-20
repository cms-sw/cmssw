#ifndef EDM_ACTIONCODES_HH
#define EDM_ACTIONCODES_HH

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>
#include <map>

namespace edm
{
  namespace actions
  {
    enum ActionCodes
      {
	IgnoreCompletely=0,
	Rethrow,
	SkipEvent,
	FailModule,
	FailPath,
	LastCode
      };

    const char* actionName(ActionCodes code);
  }

  class ActionTable
  {
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

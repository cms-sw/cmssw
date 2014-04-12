#ifndef FWCore_Framework_ExceptionActions_h
#define FWCore_Framework_ExceptionActions_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>
#include <map>

namespace edm {
  namespace exception_actions {
    enum ActionCodes {
      IgnoreCompletely=0,
      Rethrow,
      SkipEvent,
      FailPath,
      LastCode
    };

    const char* actionName(ActionCodes code);
  }

  class ExceptionToActionTable {
  public:
    typedef std::map<std::string, exception_actions::ActionCodes> ActionMap;

    ExceptionToActionTable();
    explicit ExceptionToActionTable(const ParameterSet&);
    ~ExceptionToActionTable();

    void add(const std::string& category, exception_actions::ActionCodes code);
    exception_actions::ActionCodes find(const std::string& category) const;

  private:
    void addDefaults();
    ActionMap map_;
  };
}
#endif

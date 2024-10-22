#ifndef L1Trigger_L1TCommon_l1t_TriggerSystem_h
#define L1Trigger_L1TCommon_l1t_TriggerSystem_h

#include <vector>
#include <string>
#include <map>
#include <set>

#include "Parameter.h"
#include "L1Trigger/L1TCommon/interface/Mask.h"

namespace l1t {

  class TriggerSystem {
  private:
    std::string sysId;

    std::map<std::string, std::string> procToRole;               // map of processors to their roles
    std::map<std::string, std::string> procToSlot;               // map of processors to their slots in some crate
    std::map<std::string, bool> procEnabled;                     // processor is active/disabled (also including daqttc)
    std::map<std::string, std::string> daqttcToRole;             // map of DAQ/TTC boards to their roles
    std::map<std::string, std::string> daqttcToCrate;            // map of DAQ/TTC boards to the crates they sit in
    std::map<std::string, std::set<std::string> > roleForProcs;  // map of roles, each describing a set of processors
    std::map<std::string, std::set<std::string> > crateForProcs;  // map of crates, each containing a set of processors
    std::map<std::string, std::set<std::string> >
        roleForDaqttcs;  // map of roles, each describing a set of DAQ/TTC boards

    std::map<std::string, std::map<std::string, Parameter> >
        procParameters;  // setting objects found in the configuration for a given processor
    std::map<std::string, std::map<std::string, Mask> >
        procMasks;  // mask objects found in the configuration for a given processor

    bool isConfigured;           // lock allowing access to the system
    mutable std::ostream *logs;  // print processing logs unless is set to a null pointer

  public:
    void configureSystemFromFiles(const char *hwCfgFile, const char *topCfgFile, const char *key);

    void addProcessor(const char *processor,
                      const char *role,
                      const char *crate,
                      const char *slot);  // must have all parameters

    void addDaq(const char *daq, const char *role, const char *crate);

    void addParameter(
        const char *id, const char *procOrRole, const char *type, const char *value, const char *delim = ",");

    void addTable(const char *id,
                  const char *procOrRole,
                  const char *columns,
                  const char *types,
                  const std::vector<std::string> &rows,
                  const char *delim);

    void addMask(const char *id, const char *procOrRoleOrDaq);

    void disableProcOrRoleOrDaq(const char *procOrRoleOrDaq);

    const std::map<std::string, std::string> &getProcToRoleAssignment(void) const noexcept { return procToRole; }
    const std::map<std::string, std::set<std::string> > &getRoleToProcsAssignment(void) const noexcept {
      return roleForProcs;
    }

    const std::map<std::string, Parameter> &getParameters(const char *processor) const;
    const std::map<std::string, Mask> &getMasks(const char *processor) const;

    bool isMasked(const char *proccessor, const char *id) const;
    bool isProcEnabled(const char *proccessor) const;

    std::string systemId(void) const noexcept { return sysId; }
    void setSystemId(const char *id) noexcept { sysId = id; }

    void setConfigured(bool state = true) noexcept { isConfigured = state; }
    void setLogStream(std::ostream *s) const noexcept { logs = s; }

    TriggerSystem(void) {
      isConfigured = false;
      logs = nullptr;
    }

    ~TriggerSystem(void) {}
  };

}  // namespace l1t

#endif

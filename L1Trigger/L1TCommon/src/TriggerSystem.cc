#include "L1Trigger/L1TCommon/interface/TriggerSystem.h"
#include "L1Trigger/L1TCommon/interface/XmlConfigParser.h"

using namespace std;

namespace l1t {

  void TriggerSystem::configureSystemFromFiles(const char *hwCfgFile, const char *topCfgFile, const char *key) {
    // read hw description xml
    // this will set the sysId
    {
      XmlConfigParser xmlRdr;
      xmlRdr.readDOMFromFile(hwCfgFile);
      xmlRdr.readRootElement(*this);
    }
    // read configuration xml files
    {
      XmlConfigParser xmlRdr;
      xmlRdr.readDOMFromFile(topCfgFile);
      xmlRdr.buildGlobalDoc(key, topCfgFile);
      xmlRdr.readContexts(key, sysId, *this);
    }
    isConfigured = true;
  }

  void TriggerSystem::addProcessor(const char *processor, const char *role, const char *crate, const char *slot) {
    // every processor must have a single defined role
    auto p2r = procToRole.find(processor);
    if (p2r != procToRole.end() && p2r->second != role)
      throw std::runtime_error("Processor: '" + string(processor) + "' already exists but with different role: '" +
                               p2r->second + "'");
    else {
      procEnabled[processor] = true;
      procToRole[processor] = role;
      procToSlot[processor] = slot;
      procParameters.insert(make_pair(string(processor), std::map<std::string, Parameter>()));
      procMasks.insert(make_pair(string(processor), std::map<std::string, Mask>()));
      roleForProcs[role].insert(processor);
      crateForProcs[crate].insert(processor);
    }
  }

  void TriggerSystem::addDaq(const char *daq, const char *role, const char *crate) {
    auto d2r = daqttcToRole.find(daq);
    if (d2r != daqttcToRole.end() && d2r->second != role)
      throw runtime_error("DAQttc: '" + string(daq) + "' already exists but with different role: " + d2r->second);
    else {
      daqttcToRole[daq] = role;
      daqttcToCrate[daq] = crate;
      roleForDaqttcs[role].insert(daq);
    }
  }

  void TriggerSystem::addParameter(
      const char *id, const char *procOrRole, const char *type, const char *value, const char *delim) {
    // some tables forget to specify delimeter and we get an empty string here
    //  force the "," default delimeter
    if (strlen(delim) == 0)
      delim = ",";

    // first try to locate a processor with name matching the procOrRole argument
    auto processor = procParameters.find(procOrRole);
    if (processor != procParameters.end()) {
      // processor found -> apply settings to this processor
      auto setting = processor->second.find(id);
      if (setting != processor->second.end()) {
        // setting with this id already exists -> always take the latest value
        // if( logs )
        //    *logs << "Warning: overriding already existing " << id
        //          << " = (" << setting->second.getType() << ") " << setting->second.getValue()
        //          << " with new value (" << type << ") " << value << endl;
        setting->second = Parameter(id, procOrRole, type, value, delim);
      } else
        processor->second.insert(make_pair(string(id), Parameter(id, procOrRole, type, value, delim)));
      // let's run a consistency check
      auto p2r = procToRole.find(procOrRole);
      if (p2r == procToRole.end())
        if (logs)
          *logs << "Warning: TriggerSystem object doesn't yet assign "
                << " a role to the processor " << procOrRole << endl;
      return;
    }

    // if we've got here, the procOrRole argument must have meaning of role,
    //  throw exception otherwise
    auto role = roleForProcs.find(procOrRole);
    if (role != roleForProcs.end()) {
      // apply setting on all of the processors for this role
      for (auto &proc : role->second) {
        auto processor = procParameters.find(proc);
        if (processor != procParameters.end()) {
          // processor found -> apply settings to this processor
          //  unless the setting with such id already exists
          auto setting = processor->second.find(id);
          if (setting == processor->second.end())
            processor->second.insert(make_pair(string(id), Parameter(id, procOrRole, type, value, delim)));
        } else {
          map<string, Parameter> tmp;
          tmp.insert(make_pair(id, Parameter(id, procOrRole, type, value)));
          procParameters.insert(make_pair(proc, std::move(tmp)));
          // construct with brace-initialization, although better looking, cannot use move constructor:
          //procParameters.insert(
          //   make_pair(proc, map<string,Parameter>( {{id,Parameter(id,procOrRole,type,value)}} ) )
          //);
        }
      }
    } else
      throw runtime_error("Processor or Role '" + string(procOrRole) + "' was not found");
  }

  void TriggerSystem::addTable(const char *id,
                               const char *procOrRole,
                               const char *columns,
                               const char *types,
                               const vector<string> &rows,
                               const char *delim) {
    // some tables forget to specify delimeter and we get an empty string here
    //  force the "," default delimeter
    if (strlen(delim) == 0)
      delim = ",";

    // first try to locate a processor with name matching the procOrRole argument
    auto processor = procParameters.find(procOrRole);
    if (processor != procParameters.end()) {
      // processor found -> apply settings to this processor
      auto setting = processor->second.find(id);
      if (setting != processor->second.end())
        // setting with this id already exists -> always take latest value
        setting->second = Parameter(id, procOrRole, types, columns, rows, delim);
      else
        processor->second.insert(make_pair(string(id), Parameter(id, procOrRole, types, columns, rows, delim)));
      // let's run a consistency check
      auto p2r = procToRole.find(procOrRole);
      if (p2r == procToRole.end())
        if (logs)
          *logs << "Warning: TriggerSystem object doesn't yet assign "
                << " a role to the processor " << procOrRole << endl;
      return;
    }

    // if we've got here, the procOrRole argument must have meaning of role,
    //  throw exception otherwise
    auto role = roleForProcs.find(procOrRole);
    if (role != roleForProcs.end()) {
      // apply setting on all of the processors for this role
      for (auto &proc : role->second) {
        auto processor = procParameters.find(proc);
        if (processor != procParameters.end()) {
          // processor found -> apply settings to this processor
          //  unless the setting with such id already exists
          auto setting = processor->second.find(id);
          if (setting == processor->second.end())
            processor->second.insert(make_pair(string(id), Parameter(id, procOrRole, types, columns, rows, delim)));
        } else {
          map<string, Parameter> tmp;
          tmp.insert(make_pair(id, Parameter(id, procOrRole, types, columns, rows, delim)));
          procParameters.insert(make_pair(proc, std::move(tmp)));
          // construct with brace-initialization, although better looking, cannot use move constructor:
          //procParameters.insert(
          //   make_pair(proc, map<string,Parameter>( {{id,Parameter(id,procOrRole,types,columns,rows,delim)}} ) )
          //);
        }
      }
    } else
      throw runtime_error("Processor or Role '" + string(procOrRole) + "' was not found");
  }

  const map<string, Parameter> &TriggerSystem::getParameters(const char *p) const {
    if (!isConfigured)
      throw runtime_error("TriggerSystem is not configured yet. First call the configureSystem method");

    auto processor = procParameters.find(p);
    if (processor == procParameters.end())
      throw runtime_error("Processor '" + string(p) + "' was not found in the configuration");

    return processor->second;
  }

  void TriggerSystem::addMask(const char *id, const char *procOrRoleOrDaq) {
    // first try to locate a processor with name matching the procOrRoleOrDaq argument
    auto processor = procMasks.find(procOrRoleOrDaq);
    if (processor != procMasks.end()) {
      // processor found -> apply settings to this processor
      auto mask = processor->second.find(id);
      if (mask != processor->second.end()) {
        // setting with this id already exists -> always take the latest value
        //if( logs)
        //   *logs << "Warning: overriding already existing " << id
        //         << " = (" << setting->second.getType() << ") " << setting->second.getValue()
        //         << " with new value (" << type << ") " << value << endl;
        mask->second = Mask(id, procOrRoleOrDaq);
      } else
        processor->second.insert(make_pair(string(id), Mask(id, procOrRoleOrDaq)));
      // let's run a consistency check
      auto p2r = procToRole.find(procOrRoleOrDaq);
      if (p2r == procToRole.end())
        if (logs)
          *logs << "Warning: TriggerSystem object doesn't yet assign "
                << " a role to the processor " << procOrRoleOrDaq << endl;
      return;
    }

    // if we've got here, the procOrRoleOrDaq argument may have meaning of role
    auto role = roleForProcs.find(procOrRoleOrDaq);
    if (role != roleForProcs.end()) {
      // apply setting on all of the processors for this role
      for (auto &proc : role->second) {
        auto processor = procMasks.find(proc);
        if (processor != procMasks.end()) {
          // processor found -> apply settings to this processor
          //  unless the setting with such id already exists
          auto mask = processor->second.find(id);
          if (mask == processor->second.end())
            processor->second.insert(make_pair(string(id), Mask(id, procOrRoleOrDaq)));
        } else {
          // here, copy constructor creates no overhead over the move constructor, whould that be defined
          procMasks.insert(make_pair(proc, map<string, Mask>({{id, Mask(id, procOrRoleOrDaq)}})));
        }
      }
      return;
    }

    // if we've got here, the procOrRoleOrDaq argument the only choise left is daqttc configuration
    //  that, in turn, can again be a daqttc processor or daqttc role
    // in either case, for daq we do not have any independent configuration apart from the mask status
    //  and the slot location of the processor that is masked (sitting, of course, in the same crate)
    auto d2c = daqttcToCrate.find(procOrRoleOrDaq);
    if (d2c != daqttcToCrate.end()) {
      // now, as we know crate of this daqttc, look for the crate's processors that match the id name
      size_t idLen = strlen(id);
      string slot = id + (idLen > 2 ? idLen - 2 : 0);  // last two digits of the port comptise the slot #
      auto processors = crateForProcs.find(d2c->second);
      if (processors != crateForProcs.end()) {
        for (auto &proc : processors->second)
          if (procToSlot[proc] == slot)
            procEnabled[proc] = false;
      } else if (logs)
        *logs << "Warning: no processors in daqttc crate for " << procOrRoleOrDaq << " ... do nothing" << endl;
      return;
    }

    // so, finally, this is daqttc role
    auto r2d = roleForDaqttcs.find(procOrRoleOrDaq);
    if (r2d != roleForDaqttcs.end()) {
      for (auto &daq : r2d->second) {
        auto processors = crateForProcs.find(daq);
        if (processors != crateForProcs.end()) {
          for (auto &proc : processors->second)
            procEnabled[proc] = false;
        } else if (logs)
          *logs << "Warning: no processors in daqttc crate " << d2c->second << " for " << procOrRoleOrDaq
                << " ... do nothing" << endl;
        return;
      }
    }

    // if we ever reach here, we've ran out of options
    throw runtime_error("Processor/DAQ or Role '" + string(procOrRoleOrDaq) + "' was not found in the map for masking");
  }

  const map<string, Mask> &TriggerSystem::getMasks(const char *p) const {
    if (!isConfigured)
      throw std::runtime_error("TriggerSystem is not configured yet. First call the configureSystem method");

    auto processor = procMasks.find(p);
    if (processor == procMasks.end())
      throw std::runtime_error("Processor '" + string(p) + "' was not found in the configuration");

    return processor->second;
  }

  bool TriggerSystem::isMasked(const char *p, const char *id) const {
    const std::map<std::string, Mask> &m = getMasks(p);

    auto mask = m.find(id);
    if (mask == m.end())
      return false;

    return true;
  }

  void TriggerSystem::disableProcOrRoleOrDaq(const char *procOrRoleOrDaq) {
    // follow the standard search steps to identify if the argument is processor or role or daqttc processor/role

    // the argument is simply a processor's name
    auto processor = procEnabled.find(procOrRoleOrDaq);
    if (processor != procEnabled.end()) {
      processor->second = false;
      return;
    }

    // role
    auto role = roleForProcs.find(procOrRoleOrDaq);
    if (role != roleForProcs.end()) {
      // apply setting on all of the processors for this role
      for (auto &proc : role->second)
        // by design procEnabled must have every single processor for every role
        procEnabled[proc] = false;
      return;
    }

    // the whole daq of a crate is disables -> disable all of the processors in the crate
    auto d2c = daqttcToCrate.find(procOrRoleOrDaq);
    if (d2c != daqttcToCrate.end()) {
      auto processors = crateForProcs.find(d2c->second);
      if (processors != crateForProcs.end()) {
        for (auto &proc : processors->second)
          // by design procEnabled must have every single processor for every crate
          procEnabled[proc] = false;
      } else if (logs)
        *logs << "Warning: no processors in daqttc crate for " << procOrRoleOrDaq << " ... do nothing" << endl;
      return;
    }

    // so, finally, this is daqttc role
    auto r2d = roleForDaqttcs.find(procOrRoleOrDaq);
    if (r2d != roleForDaqttcs.end()) {
      for (auto &daq : r2d->second) {
        auto d2c = daqttcToCrate.find(daq);
        if (d2c != daqttcToCrate.end()) {
          auto processors = crateForProcs.find(d2c->second);
          if (processors != crateForProcs.end()) {
            for (auto &proc : processors->second)
              procEnabled[proc] = false;
          } else if (logs)
            *logs << "Warning: no processors in daqttc crate " << d2c->second << " for " << procOrRoleOrDaq
                  << " ... do nothing" << endl;
        } else if (logs)
          *logs << "Warning: daqttc " << daq << " has no crate "
                << " ... do nothing" << endl;
        return;
      }
    }

    // if we ever reach here, we've ran out of options
    throw runtime_error("Processor/DAQ or Role '" + string(procOrRoleOrDaq) + "' was not found");
  }

  bool TriggerSystem::isProcEnabled(const char *p) const {
    if (!isConfigured)
      throw std::runtime_error("TriggerSystem is not configured yet. First call the configureSystem method");

    auto processor = procEnabled.find(p);
    if (processor == procEnabled.end())
      throw runtime_error("Processor '" + string(p) + "' not found");

    return processor->second;
  }

}  // namespace l1t

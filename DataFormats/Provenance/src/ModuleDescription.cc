#include "DataFormats/Provenance/interface/ModuleDescription.h"

#include <ostream>
#include <sstream>
#include <limits>
#include <atomic>

static std::atomic<unsigned int> s_id{0};

/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

namespace edm {

  ModuleDescription::ModuleDescription() :
    parameterSetID_(),
    moduleName_(),
    moduleLabel_(),
    processConfigurationPtr_(nullptr),
    id_(invalidID()){}

  ModuleDescription::ModuleDescription(
		ParameterSetID const& pid,
		std::string const& modName,
		std::string const& modLabel) : ModuleDescription{pid, modName, modLabel, nullptr, invalidID()} {}

  ModuleDescription::ModuleDescription(
		ParameterSetID const& pid,
		std::string const& modName,
		std::string const& modLabel,
		ProcessConfiguration const* procConfig,
    unsigned int iID) :
			parameterSetID_(pid),
			moduleName_(modName),
			moduleLabel_(modLabel),
			processConfigurationPtr_(procConfig),
      id_(iID){}

  ModuleDescription::ModuleDescription(
		std::string const& modName,
		std::string const& modLabel) : ModuleDescription{ParameterSetID(), modName, modLabel, nullptr, invalidID()} {}

  ModuleDescription::ModuleDescription(
		std::string const& modName,
		std::string const& modLabel,
		ProcessConfiguration const* procConfig) : ModuleDescription{ParameterSetID(), modName, modLabel, procConfig, invalidID()} {}

  ModuleDescription::~ModuleDescription() {}

  ProcessConfiguration const&
  ModuleDescription::processConfiguration() const {
    return *processConfigurationPtr_;
  }

  std::string const&
  ModuleDescription::processName() const {
    return processConfiguration().processName();
  }

  std::string const&
  ModuleDescription::releaseVersion() const {
    return processConfiguration().releaseVersion();
  }

  std::string const&
  ModuleDescription::passID() const {
    return processConfiguration().passID();
  }

  ParameterSetID const&
  ModuleDescription::mainParameterSetID() const {
    return processConfiguration().parameterSetID();
  }

  bool
  ModuleDescription::operator<(ModuleDescription const& rh) const {
    if (moduleLabel() < rh.moduleLabel()) return true;
    if (rh.moduleLabel() < moduleLabel()) return false;
    if (processName() < rh.processName()) return true;
    if (rh.processName() < processName()) return false;
    if (moduleName() < rh.moduleName()) return true;
    if (rh.moduleName() < moduleName()) return false;
    if (parameterSetID() < rh.parameterSetID()) return true;
    if (rh.parameterSetID() < parameterSetID()) return false;
    if (releaseVersion() < rh.releaseVersion()) return true;
    if (rh.releaseVersion() < releaseVersion()) return false;
    if (passID() < rh.passID()) return true;
    return false;
  }

  bool
  ModuleDescription::operator==(ModuleDescription const& rh) const {
    return !((*this) < rh || rh < (*this));
  }

  bool
  ModuleDescription::operator!=(ModuleDescription const& rh) const {
    return !((*this) == rh);
  }

  unsigned int
  ModuleDescription::getUniqueID() {
    return s_id++;
  }

  void
  ModuleDescription::write(std::ostream& os) const {
    os  << "Module type=" << moduleName() << ", "
	<< "Module label=" << moduleLabel() << ", "
	<< "Parameter Set ID=" << parameterSetID();
	//<< "Parameter Set ID=" << parameterSetID() << ", "
	//<< "Process name=" << processName() << ", "
	//<< "Release Version=" << releaseVersion() << ", "
	//<< "Pass ID=" << passID() << ", "
	//<< "Main Parameter Set ID=" << mainParameterSetID();
  }
}

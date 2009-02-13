#include "DataFormats/Provenance/interface/ModuleDescription.h"

#include "FWCore/Utilities/interface/Digest.h"

#include <ostream>
#include <sstream>

/*----------------------------------------------------------------------

$Id: ModuleDescription.cc,v 1.3 2008/12/20 18:40:28 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {

  ModuleDescription::ModuleDescription() :
    parameterSetID_(),
    moduleName_(),
    moduleLabel_(),
    processConfigurationPtr_(new ProcessConfiguration()) {}

  ModuleDescription::ModuleDescription(
		ParameterSetID const& pid,
		std::string const& modName,
		std::string const& modLabel) :
			parameterSetID_(pid),
			moduleName_(modName),
			moduleLabel_(modLabel),
			processConfigurationPtr_(new ProcessConfiguration()) {}

  ModuleDescription::ModuleDescription(
		ParameterSetID const& pid,
		std::string const& modName,
		std::string const& modLabel,
		boost::shared_ptr<ProcessConfiguration> procConfig) :
			parameterSetID_(pid),
			moduleName_(modName),
			moduleLabel_(modLabel),
			processConfigurationPtr_(procConfig) {}

  ModuleDescription::ModuleDescription(
		std::string const& modName,
		std::string const& modLabel) :
			parameterSetID_(),
			moduleName_(modName),
			moduleLabel_(modLabel),
			processConfigurationPtr_(new ProcessConfiguration()) {}

  ModuleDescription::ModuleDescription(
		std::string const& modName,
		std::string const& modLabel,
		boost::shared_ptr<ProcessConfiguration> procConfig) :
			parameterSetID_(),
			moduleName_(modName),
			moduleLabel_(modLabel),
			processConfigurationPtr_(procConfig) {}

  ProcessConfiguration const&
  ModuleDescription::processConfiguration() const {
    return *processConfigurationPtr_;
  }

  ProcessConfigurationID
  ModuleDescription::processConfigurationID() const {
    return processConfiguration().id();
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

  ModuleDescriptionID
  ModuleDescription::id() const
  {
    // This implementation is ripe for optimization.
    // We do not use operator<< because it does not write out everything.
    std::ostringstream oss;
    oss << parameterSetID() << ' ' 
	<< moduleName() << ' '
	<< moduleLabel() << ' '
	<< mainParameterSetID() << ' '
	<< releaseVersion() << ' '
	<< processName() << ' '
	<< passID();
    std::string stringrep = oss.str();
    cms::Digest md5alg(stringrep);
    return ModuleDescriptionID(md5alg.digest().toString());
  }

}

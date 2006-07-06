#ifndef Common_ModuleDescription_h
#define Common_ModuleDescription_h

/*----------------------------------------------------------------------
  
ModuleDescription: The description of a producer module.

$Id: ModuleDescription.h,v 1.2.2.4 2006/07/05 23:55:14 wmtan Exp $
----------------------------------------------------------------------*/
#include <string>
#include <iostream>

#include "DataFormats/Common/interface/PassID.h"
#include "DataFormats/Common/interface/ParameterSetID.h"
#include "DataFormats/Common/interface/ProcessConfiguration.h"
#include "DataFormats/Common/interface/ReleaseVersion.h"
#include "DataFormats/Common/interface/ModuleDescriptionID.h"

namespace edm {

  // once a module is born, these parts of the module's product provenance
  // are constant   (change to ModuleDescription)

  struct ModuleDescription {

    ModuleDescription();

    ParameterSetID const& parameterSetID() const {return parameterSetID_;}
    std::string const& moduleName() const {return moduleName_;}
    std::string const& moduleLabel() const {return moduleLabel_;}
    ProcessConfiguration const& processConfiguration() const {return processConfiguration_;}
    std::string const& processName() const {return processConfiguration().processName();}
    std::string const& releaseVersion() const {return processConfiguration().releaseVersion();}
    std::string const& passID() const {return processConfiguration().passID();}
    ParameterSetID const& mainParameterSetID() const {return processConfiguration().parameterSetID();}

    // compiler-written copy c'tor, assignment, and d'tor are correct.

    bool operator<(ModuleDescription const& rh) const;

    bool operator==(ModuleDescription const& rh) const;

    bool operator!=(ModuleDescription const& rh) const;
    
    ModuleDescriptionID id() const;

    // ID of parameter set of the creator
    ParameterSetID parameterSetID_;

    // The class name of the creator
    std::string moduleName_;    

    // A human friendly string that uniquely identifies the EDProducer
    // and becomes part of the identity of a product that it produces
    std::string moduleLabel_;

    // The process configuration.
    ProcessConfiguration processConfiguration_;
  };

  inline
  std::ostream& operator<<(std::ostream& ost, const ModuleDescription& md) {
    ost << "Module type=" << md.moduleName() << ", "
	<< "Module label=" << md.moduleLabel() << ", "
	<< "Process name=" << md.processConfiguration().processName();
    return ost;
  }
}
#endif

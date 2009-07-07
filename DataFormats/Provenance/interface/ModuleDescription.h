#ifndef DataFormats_Provenance_ModuleDescription_h
#define DataFormats_Provenance_ModuleDescription_h

/*----------------------------------------------------------------------
  
ModuleDescription: The description of a producer module.

----------------------------------------------------------------------*/
#include <string>
#include <iosfwd>
#include "boost/shared_ptr.hpp"

#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProcessConfigurationID.h"

namespace edm {

  // once a module is born, these parts of the module's product provenance
  // are constant   (change to ModuleDescription)

  class ModuleDescription {
  public:

    ModuleDescription();

    ModuleDescription(std::string const& modName,
		      std::string const& modLabel);

    ModuleDescription(std::string const& modName,
		      std::string const& modLabel,
		      boost::shared_ptr<ProcessConfiguration> procConfig);

    ModuleDescription(ParameterSetID const& pid,
		      std::string const& modName,
		      std::string const& modLabel);

    ModuleDescription(ParameterSetID const& pid,
		      std::string const& modName,
		      std::string const& modLabel,
		      boost::shared_ptr<ProcessConfiguration> procConfig);

    ~ModuleDescription();

    void write(std::ostream& os) const;

    ParameterSetID const& parameterSetID() const {return parameterSetID_;}
    std::string const& moduleName() const {return moduleName_;}
    std::string const& moduleLabel() const {return moduleLabel_;}
    ProcessConfiguration const& processConfiguration() const;
    ProcessConfigurationID processConfigurationID() const;
    std::string const& processName() const;
    std::string const& releaseVersion() const;
    std::string const& passID() const;
    ParameterSetID const& mainParameterSetID() const;

    // compiler-written copy c'tor, assignment, and d'tor are correct.

    bool operator<(ModuleDescription const& rh) const;

    bool operator==(ModuleDescription const& rh) const;

    bool operator!=(ModuleDescription const& rh) const;
    
  // private: public for now.  Will be made private ASAP.
    // ID of parameter set of the creator
    ParameterSetID parameterSetID_;

    // The class name of the creator
    std::string moduleName_;    

    // A human friendly string that uniquely identifies the EDProducer
    // and becomes part of the identity of a product that it produces
    std::string moduleLabel_;

    // The process configuration.
    boost::shared_ptr<ProcessConfiguration> processConfigurationPtr_;
  };

  inline
  std::ostream&
  operator<<(std::ostream& os, const ModuleDescription& p) {
    p.write(os);
    return os;
  }

}
#endif

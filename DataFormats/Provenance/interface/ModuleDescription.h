#ifndef DataFormats_Provenance_ModuleDescription_h
#define DataFormats_Provenance_ModuleDescription_h

/*----------------------------------------------------------------------

ModuleDescription: The description of a producer module.

----------------------------------------------------------------------*/
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"

#include "boost/shared_ptr.hpp"

#include <iosfwd>
#include <string>

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
                      ProcessConfiguration const* procConfig);

    ModuleDescription(ParameterSetID const& pid,
                      std::string const& modName,
                      std::string const& modLabel);

    ModuleDescription(ParameterSetID const& pid,
                      std::string const& modName,
                      std::string const& modLabel,
                      ProcessConfiguration const* procConfig,
                      unsigned int modID);

    ~ModuleDescription();

    void write(std::ostream& os) const;

    ParameterSetID const& parameterSetID() const {return parameterSetID_;}
    std::string const& moduleName() const {return moduleName_;}
    std::string const& moduleLabel() const {return moduleLabel_;}
    ///A unique ID for a module declared in the Process. The id is only unique for the Process and not across different Processes.
    ///If the id is invalid, will return the max unsigned int value.
    unsigned int id() const {return id_;}
    ProcessConfiguration const& processConfiguration() const;

    std::string const& processName() const;
    std::string const& releaseVersion() const;
    std::string const& passID() const;
    ParameterSetID const& mainParameterSetID() const;

    // compiler-written copy c'tor, assignment, and d'tor are correct.

    bool operator<(ModuleDescription const& rh) const;

    bool operator==(ModuleDescription const& rh) const;

    bool operator!=(ModuleDescription const& rh) const;

    ///Returns a unique id each time called. Intended to be passed to ModuleDescription's constructor's modID argument. Thread safe.
    static unsigned int getUniqueID();

  private:

    // ID of parameter set of the creator
    ParameterSetID parameterSetID_;

    // The class name of the creator
    std::string moduleName_;

    // A human friendly string that uniquely identifies the EDProducer
    // and becomes part of the identity of a product that it produces
    std::string moduleLabel_;

    // The process configuration.
    ProcessConfiguration const* processConfigurationPtr_;
    
    unsigned int id_;
  };

  inline
  std::ostream&
  operator<<(std::ostream& os, ModuleDescription const& p) {
    p.write(os);
    return os;
  }

}
#endif

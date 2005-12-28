#ifndef Framework_ModuleDescription_h
#define Framework_ModuleDescription_h

/*----------------------------------------------------------------------
  
ModuleDescription: The description of a producer module.

$Id: ModuleDescription.h,v 1.7 2005/12/13 05:35:22 wmtan Exp $
----------------------------------------------------------------------*/
#include <string>
#include <iostream>

#include "FWCore/Framework/interface/PassID.h"
#include "FWCore/EDProduct/interface/ParameterSetID.h"
#include "FWCore/Framework/interface/VersionNumber.h"

namespace edm {

  // Needed temporarily for backward compatibility.
  // Other input services still refer to PS_ID.
  typedef ParameterSetID PS_ID;

  // once a module is born, these parts of the module's product provenance
  // are constant   (change to ModuleDescription)
  struct ModuleDescription {

    // ID of parameter set of the creator
    ParameterSetID pid;

    // The class name of the creator
    std::string moduleName_;    

    // A human friendly string that uniquely identifies the EDProducer
    // and becomes part of the identity of a product that it produces
    std::string moduleLabel_;

    // the release tag of the executable
    VersionNumber versionNumber_;

    // the physical process that this program was part of (e.g. production)
    std::string processName_;

    // what the heck is this? I think its the version of the processName_
    // e.g. second production pass
    PassID pass;

    bool operator<(ModuleDescription const& rh) const;

    bool operator==(ModuleDescription const& rh) const;

    bool operator!=(ModuleDescription const& rh) const;
  };

  inline
  std::ostream& operator<<(std::ostream& ost, const ModuleDescription& md) {
    ost << "Module type=" << md.moduleName_ << ", "
	<< "Module label=" << md.moduleLabel_ << ", "
	<< "Process name=" << md.processName_;

    return ost;
  }
}
#endif

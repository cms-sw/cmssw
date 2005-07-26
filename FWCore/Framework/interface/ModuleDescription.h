#ifndef Framework_ModuleDescription_h
#define Framework_ModuleDescription_h

/*----------------------------------------------------------------------
  
ModuleDescription: The description of a producer module.

$Id: ModuleDescription.h,v 1.3 2005/07/20 03:00:36 jbk Exp $
----------------------------------------------------------------------*/
#include <string>
#include <iostream>

#include "FWCore/Framework/interface/PassID.h"
#include "FWCore/Framework/interface/PS_ID.h"
#include "FWCore/Framework/interface/VersionNumber.h"

namespace edm {

  // once a module is born, these parts of the module's product provenance
  // are constant   (change to ModuleDescription)
  struct ModuleDescription {

    // ID of parameter set of the creator
    PS_ID pid;

    // The class name of the creator
    std::string module_name;    

    // A human friendly string that uniquely identifies the EDProducer
    // and becomes part of the identity of a product that it produces
    std::string module_label;

    // the release tag of the executable
    VersionNumber version_number;

    // the physical process that this program was part of (e.g. production)
    std::string process_name;

    // what the heck is this? I think its the version of the process_name
    // e.g. second production pass
    PassID pass;

    bool operator<(ModuleDescription const& rh) const;

    bool operator==(ModuleDescription const& rh) const;
  };

  inline
  std::ostream& operator<<(std::ostream& ost, const ModuleDescription& md) {
    ost << "Module type=" << md.module_name << ", "
	<< "Module label=" << md.module_label << ", "
	<< "Process name=" << md.process_name;

    return ost;
  }
}
#endif

#ifndef CoreFramework_ModuleDescription_h
#define CoreFramework_ModuleDescription_h

/*----------------------------------------------------------------------
  
ModuleDescription: The description of a producer module.

$Id: ModuleDescription.h,v 1.0 2005/06/23 22:01:31 wmtan Exp $
----------------------------------------------------------------------*/
#include <string>

#include "FWCore/CoreFramework/interface/PassID.h"
#include "FWCore/CoreFramework/interface/PS_ID.h"
#include "FWCore/CoreFramework/interface/VersionNumber.h"

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
  };

  inline
  bool operator==(ModuleDescription const& a, ModuleDescription const& b) {
    return 
      a.pid == b.pid
      && a.module_name == b.module_name
      && a.module_label == b.module_label 
      && a.version_number == b.version_number
      && a.process_name == b.process_name
      && a.pass == b.pass;
  } 
}
#endif

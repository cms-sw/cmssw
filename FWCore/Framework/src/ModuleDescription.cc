#include "FWCore/Framework/interface/ModuleDescription.h"

/*----------------------------------------------------------------------

$Id: ModuleDescription.cc,v 1.3 2005/07/26 20:16:21 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {

  bool
  ModuleDescription::operator<(ModuleDescription const& rh) const {
    if (module_label < rh.module_label) return true;
    if (rh.module_label < module_label) return false;
    if (process_name < rh.process_name) return true;
    if (rh.process_name < process_name) return false;
    if (module_name < rh.module_name) return true;
    if (rh.module_name < module_name) return false;
    if (pid < rh.pid) return true;
    if (rh.pid < pid) return false;
    if (version_number < rh.version_number) return true;
    if (rh.version_number < version_number) return false;
    if (pass < rh.pass) return true;
    return false;
  } 

  bool
  ModuleDescription::operator==(ModuleDescription const& rh) const {
    return !((*this) < rh || rh < (*this));
  }
}

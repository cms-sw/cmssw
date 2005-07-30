#include "FWCore/Framework/interface/ModuleDescription.h"

/*----------------------------------------------------------------------

$Id: ModuleDescription.cc,v 1.1 2005/07/26 23:10:09 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {

  bool
  ModuleDescription::operator<(ModuleDescription const& rh) const {
    if (moduleLabel_ < rh.moduleLabel_) return true;
    if (rh.moduleLabel_ < moduleLabel_) return false;
    if (processName_ < rh.processName_) return true;
    if (rh.processName_ < processName_) return false;
    if (moduleName_ < rh.moduleName_) return true;
    if (rh.moduleName_ < moduleName_) return false;
    if (pid < rh.pid) return true;
    if (rh.pid < pid) return false;
    if (versionNumber_ < rh.versionNumber_) return true;
    if (rh.versionNumber_ < versionNumber_) return false;
    if (pass < rh.pass) return true;
    return false;
  } 

  bool
  ModuleDescription::operator==(ModuleDescription const& rh) const {
    return !((*this) < rh || rh < (*this));
  }
}

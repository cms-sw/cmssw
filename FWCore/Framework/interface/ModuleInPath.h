#ifndef FWCore_Framework_ModuleInPath_h
#define FWCore_Framework_ModuleInPath_h

/** Class used to hold the traits for how a module should be handled in a Path
 */

#include "FWCore/Framework/interface/WorkerInPath.h"

namespace edm {
  class ModuleDescription;

  struct ModuleInPath {
    ModuleInPath(ModuleDescription const* iDesc, WorkerInPath::FilterAction iAct, unsigned int iPlace, bool iConcurrent)
        : description_(iDesc), placeInPath_(iPlace), action_(iAct), runConcurrently_(iConcurrent) {}
    ModuleDescription const* description_;
    unsigned int placeInPath_;
    WorkerInPath::FilterAction action_;
    bool runConcurrently_;
  };
}  // namespace edm
#endif
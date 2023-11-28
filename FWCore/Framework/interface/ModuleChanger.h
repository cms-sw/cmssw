#ifndef FWCore_Framework_ModuleChanger_h
#define FWCore_Framework_ModuleChanger_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     ModuleChanger
//
/**\class ModuleChanger ModuleChanger.h FWCore/Framework/interface/ModuleChanger.h

 Description: Handles modifying a module after the job has started

 Usage:
    This class is used by an EDLooper at the end of a loop in order to modify the
 parameters of a module.

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Jul 15 15:05:17 EDT 2010
//

// system include files
#include <string>

// user include files
#include "FWCore/Utilities/interface/propagate_const.h"
#include "FWCore/Framework/interface/ESRecordsToProductResolverIndices.h"

// forward declarations

namespace edm {
  class ParameterSet;
  class Schedule;
  class ProductRegistry;

  class ModuleChanger {
  public:
    ModuleChanger(Schedule*, ProductRegistry const* iReg, eventsetup::ESRecordsToProductResolverIndices);
    ModuleChanger(const ModuleChanger&) = delete;                   // stop default
    const ModuleChanger& operator=(const ModuleChanger&) = delete;  // stop default
    virtual ~ModuleChanger();

    // ---------- const member functions ---------------------

    // ---------- static member functions --------------------

    // ---------- member functions ---------------------------
    bool changeModule(const std::string& iLabel, const ParameterSet& iPSet);

  private:
    // ---------- member data --------------------------------
    edm::propagate_const<Schedule*> schedule_;
    ProductRegistry const* registry_;
    eventsetup::ESRecordsToProductResolverIndices indices_;
  };
}  // namespace edm
#endif

#ifndef DataFormats_FWLite_RunFactory_h
#define DataFormats_FWLite_RunFactory_h
// -*- C++ -*-
//
// Package:     DataFormats/FWLite
// Class  :     RunFactory
//
/**\class RunFactory RunFactory.h src/DataFormats/interface/RunFactory.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Wed Feb 10 11:15:16 CST 2010
//

#include <memory>

#include "DataFormats/FWLite/interface/Run.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

namespace fwlite {
  class RunFactory {
  public:
    RunFactory();
    virtual ~RunFactory();

    // ---------- const member functions ---------------------
    std::shared_ptr<fwlite::Run> makeRun(std::shared_ptr<BranchMapReader> branchMap) const;

  private:
    RunFactory(const RunFactory&) = delete;  // stop default

    const RunFactory& operator=(const RunFactory&) = delete;  // stop default
    //This class is not inteded to be used across different threads
    CMS_SA_ALLOW mutable std::shared_ptr<fwlite::Run> run_;

    // ---------- member data --------------------------------
  };
}  // namespace fwlite

#endif

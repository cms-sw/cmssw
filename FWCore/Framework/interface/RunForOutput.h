#ifndef FWCore_Framework_RunForOutput_h
#define FWCore_Framework_RunForOutput_h

// -*- C++ -*-
//
// Package:     Framework
// Class  :     RunForOutput
//
/**\class RunForOutput RunForOutput.h FWCore/Framework/interface/RunForOutput.h

Description: This is the primary interface for outputting run products

For its usage, see "FWCore/Framework/interface/PrincipalGetAdapter.h"

*/
/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "FWCore/Framework/interface/OccurrenceForOutput.h"
#include "FWCore/Framework/interface/PrincipalGetAdapter.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

namespace edmtest {
  class TestOutputModule;
}

namespace edm {
  class MergeableRunProductMetadata;
  class ModuleCallingContext;
  
  class RunForOutput : public OccurrenceForOutput {
  public:
    RunForOutput(RunPrincipal const& rp, ModuleDescription const& md,
                 ModuleCallingContext const*, bool isAtEnd,
                 MergeableRunProductMetadata const* = nullptr);
    ~RunForOutput() override;

    RunAuxiliary const& runAuxiliary() const {return aux_;}
    RunID const& id() const {return aux_.id();}
    RunNumber_t run() const {return aux_.run();}
    Timestamp const& beginTime() const {return aux_.beginTime();}
    Timestamp const& endTime() const {return aux_.endTime();}
    MergeableRunProductMetadata const* mergeableRunProductMetadata() const { return mergeableRunProductMetadata_; }

  private:
    friend class edmtest::TestOutputModule; // For testing

    RunPrincipal const&
    runPrincipal() const;

    RunAuxiliary const& aux_;

    MergeableRunProductMetadata const* mergeableRunProductMetadata_;

    static const std::string emptyString_;
  };
}
#endif

#ifndef FWCore_Framework_ProcessBlockForOutput_h
#define FWCore_Framework_ProcessBlockForOutput_h

// -*- C++ -*-
//
// Package:     Framework
// Class  :     ProcessBlockForOutput
//
/**\class edm::ProcessBlockForOutput

Description: This is the primary interface for output modules
writing ProcessBlock products

\author W. David Dagenhart, created 29 October 2020

*/

#include "DataFormats/Provenance/interface/ProvenanceFwd.h"
#include "FWCore/Framework/interface/OccurrenceForOutput.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistryfwd.h"

#include <string>

namespace edm {
  class ProcessBlockPrincipal;

  class ProcessBlockForOutput : public OccurrenceForOutput {
  public:
    ProcessBlockForOutput(ProcessBlockPrincipal const&,
                          ModuleDescription const&,
                          ModuleCallingContext const*,
                          bool isAtEnd);
    ~ProcessBlockForOutput() override;

    std::string const& processName() const { return *processName_; }

  private:
    std::string const* processName_;
  };
}  // namespace edm
#endif

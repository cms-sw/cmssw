#ifndef FWCore_Framework_EventForOutput_h
#define FWCore_Framework_EventForOutput_h

// -*- C++ -*-
//
// Package:     Framework
// Class  :     EventForOutput
//
/**\class EventForOutput EventForOutputForOutput.h FWCore/Framework/interface/EventForOutputForOutput.h

*/
/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/BasicHandle.h"
#include "DataFormats/Common/interface/ConvertHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Provenance/interface/BranchListIndex.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/EventSelectionID.h"
#include "DataFormats/Provenance/interface/RunID.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/OccurrenceForOutput.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/propagate_const.h"

#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

class testEventGetRefBeforePut;

namespace edmtest {
  class TestOutputModule;
}

namespace edm {

  class BranchDescription;
  class ModuleCallingContext;
  class ProductProvenanceRetriever;
  class EDConsumerBase;

  class EventForOutput : public OccurrenceForOutput {
  public:
    EventForOutput(EventPrincipal const& ep, ModuleDescription const& md,
          ModuleCallingContext const*);
    ~EventForOutput() override;
    
    EventAuxiliary const& eventAuxiliary() const {return aux_;}
    EventID const& id() const {return aux_.id();}
    EventNumber_t event() const {return aux_.event();}
    LuminosityBlockNumber_t luminosityBlock() const {return aux_.luminosityBlock();}
    Timestamp const& time() const {return aux_.time();}
    
    ///\return The id for the particular Stream processing the Event
    StreamID streamID() const {
      return streamID_;
    }

    LuminosityBlockForOutput const&
    getLuminosityBlock() const {
      return *luminosityBlock_;
    }

    RunForOutput const&
    getRun() const;

    RunNumber_t
    run() const {return id().run();}
    
    BranchListIndexes const& branchListIndexes() const;

    EventSelectionIDVector const& eventSelectionIDs() const;

    ProductProvenanceRetriever const* productProvenanceRetrieverPtr() const;

  private:
    friend class edmtest::TestOutputModule; // For testing

    EventPrincipal const&
    eventPrincipal() const;

    EventAuxiliary const& aux_;
    std::shared_ptr<LuminosityBlockForOutput const> const luminosityBlock_;

    StreamID streamID_;
  };
}
#endif


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
#include "FWCore/Framework/interface/PrincipalGetAdapter.h"
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

  class EventForOutput {
  public:
    EventForOutput(EventPrincipal const& ep, ModuleDescription const& md,
          ModuleCallingContext const*);
    virtual ~EventForOutput();
    
    //Used in conjunction with EDGetToken
    void setConsumer(EDConsumerBase const* iConsumer);
    
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

    ProcessHistoryID const& processHistoryID() const;

    bool
    getByToken(EDGetToken token, TypeID const& typeID, BasicHandle& result) const;

    template<typename PROD>
    bool
    getByToken(EDGetToken token, Handle<PROD>& result) const;

    template<typename PROD>
    bool
    getByToken(EDGetTokenT<PROD> token, Handle<PROD>& result) const;

    Provenance
    getProvenance(BranchID const& theID) const;

    void
    getAllProvenance(std::vector<Provenance const*>& provenances) const;

    // Return true if this Event has been subjected to a process with
    // the given processName, and false otherwise.
    // If true is returned, then ps is filled with the ParameterSet
    // used to configure the identified process.
    bool
    getProcessParameterSet(std::string const& processName, ParameterSet& ps) const;

    virtual ProcessHistory const&
    processHistory() const;

    size_t size() const;

    ProductProvenanceRetriever const* productProvenanceRetrieverPtr() const;

  private:
    friend class edmtest::TestOutputModule; // For testing
    ModuleCallingContext const* moduleCallingContext() const { return moduleCallingContext_; }

    EventPrincipal const&
    eventPrincipal() const;

    PrincipalGetAdapter provRecorder_;

    EventAuxiliary const& aux_;
    std::shared_ptr<LuminosityBlockForOutput const> const luminosityBlock_;

    StreamID streamID_;
    ModuleCallingContext const* moduleCallingContext_;
  };

  template<typename PROD>
  bool
  EventForOutput::getByToken(EDGetToken token, Handle<PROD>& result) const {
    result.clear();
    BasicHandle bh = provRecorder_.getByToken_(TypeID(typeid(PROD)),PRODUCT_TYPE, token, moduleCallingContext_);
    convert_handle(std::move(bh), result);  // throws on conversion error
    if (result.failedToGet()) {
      return false;
    }
    return true;
  }
  
  template<typename PROD>
  bool
  EventForOutput::getByToken(EDGetTokenT<PROD> token, Handle<PROD>& result) const {
    result.clear();
    BasicHandle bh = provRecorder_.getByToken_(TypeID(typeid(PROD)),PRODUCT_TYPE, token, moduleCallingContext_);
    convert_handle(std::move(bh), result);  // throws on conversion error
    if (result.failedToGet()) {
      return false;
    }
    return true;
  }

}
#endif


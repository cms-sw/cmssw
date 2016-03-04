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
#include "FWCore/Framework/interface/PrincipalGetAdapter.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

namespace edmtest {
  class TestOutputModule;
}

namespace edm {
  class ModuleCallingContext;
  
  class RunForOutput {
  public:
    RunForOutput(RunPrincipal const& rp, ModuleDescription const& md,
        ModuleCallingContext const*);
    ~RunForOutput();

    //Used in conjunction with EDGetToken
    void setConsumer(EDConsumerBase const* iConsumer) {
      provRecorder_.setConsumer(iConsumer);
    }
    
    typedef PrincipalGetAdapter Base;
    RunAuxiliary const& runAuxiliary() const {return aux_;}
    RunID const& id() const {return aux_.id();}
    RunNumber_t run() const {return aux_.run();}
    Timestamp const& beginTime() const {return aux_.beginTime();}
    Timestamp const& endTime() const {return aux_.endTime();}

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

    ProcessHistoryID const& processHistoryID() const;

    ProcessHistory const&
    processHistory() const;

  private:
    friend class edmtest::TestOutputModule; // For testing
    ModuleCallingContext const* moduleCallingContext() const { return moduleCallingContext_; }

    RunPrincipal const&
    runPrincipal() const;

    PrincipalGetAdapter provRecorder_;
    RunAuxiliary const& aux_;
    ModuleCallingContext const* moduleCallingContext_;

    static const std::string emptyString_;
  };

  template<typename PROD>
  bool
  RunForOutput::getByToken(EDGetToken token, Handle<PROD>& result) const {
    if(!provRecorder_.checkIfComplete<PROD>()) {
      principal_get_adapter_detail::throwOnPrematureRead("Run", TypeID(typeid(PROD)), token);
    }
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
  RunForOutput::getByToken(EDGetTokenT<PROD> token, Handle<PROD>& result) const {
    if(!provRecorder_.checkIfComplete<PROD>()) {
      principal_get_adapter_detail::throwOnPrematureRead("Run", TypeID(typeid(PROD)), token);
    }
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

#ifndef FWCore_Framework_LuminosityBlockForOutput_h
#define FWCore_Framework_LuminosityBlockForOutput_h

// -*- C++ -*-
//
// Package:     Framework
// Class  :     LuminosityBlockForOutput
//
/**\class LuminosityBlockForOutput LuminosityBlockForOutput.h FWCore/Framework/interface/LuminosityBlockForOutput.h

Description: This is the primary interface for accessing per luminosity block EDProducts
and inserting new derived per luminosity block EDProducts.

For its usage, see "FWCore/Framework/interface/PrincipalGetAdapter.h"

*/
/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/PrincipalGetAdapter.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/propagate_const.h"

#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

namespace edmtest {
  class TestOutputModule;
}

namespace edm {
  class ModuleCallingContext;
  
  class LuminosityBlockForOutput {
  public:
    LuminosityBlockForOutput(LuminosityBlockPrincipal const& lbp, ModuleDescription const& md,
                    ModuleCallingContext const*);
    ~LuminosityBlockForOutput();

    LuminosityBlockAuxiliary const& luminosityBlockAuxiliary() const {return aux_;}
    LuminosityBlockID const& id() const {return aux_.id();}
    LuminosityBlockNumber_t luminosityBlock() const {return aux_.luminosityBlock();}
    RunNumber_t run() const {return aux_.run();}
    Timestamp const& beginTime() const {return aux_.beginTime();}
    Timestamp const& endTime() const {return aux_.endTime();}

    //Used in conjunction with EDGetToken
    void setConsumer(EDConsumerBase const* iConsumer);
    
    bool
    getByToken(EDGetToken token, TypeID const& typeID, BasicHandle& result) const;

    template<typename PROD>
    bool
    getByToken(EDGetToken token, Handle<PROD>& result) const;
    
    template<typename PROD>
    bool
    getByToken(EDGetTokenT<PROD> token, Handle<PROD>& result) const;

    RunForOutput const&
    getRun() const {
      return *run_;
    }

    Provenance
    getProvenance(BranchID const& theID) const;

    void
    getAllProvenance(std::vector<Provenance const*>& provenances) const;

    ProcessHistoryID const& processHistoryID() const;

    ProcessHistory const&
    processHistory() const;

  private:
    ModuleCallingContext const* moduleCallingContext() const { return moduleCallingContext_; }
    friend class edmtest::TestOutputModule; // For testing

    LuminosityBlockPrincipal const&
    luminosityBlockPrincipal() const;

    PrincipalGetAdapter provRecorder_;
    LuminosityBlockAuxiliary const& aux_;
    std::shared_ptr<RunForOutput const> const run_;
    ModuleCallingContext const* moduleCallingContext_;
  };

  template<typename PROD>
  bool
  LuminosityBlockForOutput::getByToken(EDGetToken token, Handle<PROD>& result) const {
    if(!provRecorder_.checkIfComplete<PROD>()) {
      principal_get_adapter_detail::throwOnPrematureRead("Lumi", TypeID(typeid(PROD)), token);
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
  LuminosityBlockForOutput::getByToken(EDGetTokenT<PROD> token, Handle<PROD>& result) const {
    if(!provRecorder_.checkIfComplete<PROD>()) {
      principal_get_adapter_detail::throwOnPrematureRead("Lumi", TypeID(typeid(PROD)), token);
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

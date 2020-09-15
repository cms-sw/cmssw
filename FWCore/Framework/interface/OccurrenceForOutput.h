#ifndef FWCore_Framework_OccurrenceForOutput_h
#define FWCore_Framework_OccurrenceForOutput_h

// -*- C++ -*-
//
// Package:     Framework
// Class  :     OccurrenceForOutput
//
/**\class edm::OccurrenceForOutput

*/
/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/BasicHandle.h"
#include "DataFormats/Common/interface/ConvertHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Provenance/interface/BranchListIndex.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/PrincipalGetAdapter.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
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
  class Principal;
  class EDConsumerBase;

  class OccurrenceForOutput {
  public:
    OccurrenceForOutput(Principal const& ep, ModuleDescription const& md, ModuleCallingContext const*, bool isAtEnd);
    virtual ~OccurrenceForOutput();

    //Used in conjunction with EDGetToken
    void setConsumer(EDConsumerBase const* iConsumer);

    ProcessHistoryID const& processHistoryID() const;

    BasicHandle getByToken(EDGetToken token, TypeID const& typeID) const;

    template <typename PROD>
    bool getByToken(EDGetToken token, Handle<PROD>& result) const;

    template <typename PROD>
    bool getByToken(EDGetTokenT<PROD> token, Handle<PROD>& result) const;

    template <typename PROD>
    Handle<PROD> getHandle(EDGetTokenT<PROD> token) const;

    Provenance getProvenance(BranchID const& theID) const;

    void getAllProvenance(std::vector<Provenance const*>& provenances) const;

    void getAllStableProvenance(std::vector<StableProvenance const*>& provenances) const;

    virtual ProcessHistory const& processHistory() const;

    size_t size() const;

  protected:
    Principal const& principal() const;

  private:
    friend class edmtest::TestOutputModule;  // For testing
    ModuleCallingContext const* moduleCallingContext() const { return moduleCallingContext_; }

    PrincipalGetAdapter provRecorder_;

    ModuleCallingContext const* moduleCallingContext_;
  };

  template <typename PROD>
  bool OccurrenceForOutput::getByToken(EDGetToken token, Handle<PROD>& result) const {
    if (!provRecorder_.checkIfComplete<PROD>()) {
      principal_get_adapter_detail::throwOnPrematureRead("RunOrLumi", TypeID(typeid(PROD)), token);
    }
    BasicHandle bh = provRecorder_.getByToken_(TypeID(typeid(PROD)), PRODUCT_TYPE, token, moduleCallingContext_);
    result = convert_handle<PROD>(std::move(bh));  // throws on conversion error
    if (result.failedToGet()) {
      return false;
    }
    return true;
  }

  template <typename PROD>
  bool OccurrenceForOutput::getByToken(EDGetTokenT<PROD> token, Handle<PROD>& result) const {
    if (!provRecorder_.checkIfComplete<PROD>()) {
      principal_get_adapter_detail::throwOnPrematureRead("RunOrLumi", TypeID(typeid(PROD)), token);
    }
    BasicHandle bh = provRecorder_.getByToken_(TypeID(typeid(PROD)), PRODUCT_TYPE, token, moduleCallingContext_);
    result = convert_handle<PROD>(std::move(bh));  // throws on conversion error
    if (result.failedToGet()) {
      return false;
    }
    return true;
  }

  template <typename PROD>
  Handle<PROD> OccurrenceForOutput::getHandle(EDGetTokenT<PROD> token) const {
    if (!provRecorder_.checkIfComplete<PROD>()) {
      principal_get_adapter_detail::throwOnPrematureRead("RunOrLumi", TypeID(typeid(PROD)), token);
    }
    BasicHandle bh = provRecorder_.getByToken_(TypeID(typeid(PROD)), PRODUCT_TYPE, token, moduleCallingContext_);
    return convert_handle<PROD>(std::move(bh));  // throws on conversion error
  }

}  // namespace edm
#endif

#ifndef FWCore_Common_EventBase_h
#define FWCore_Common_EventBase_h
// -*- C++ -*-
//
// Package:     FWCore/Common
// Class  :     EventBase
//
/**\class EventBase EventBase.h FWCore/Common/interface/EventBase.h

 Description: Base class for Events in both the full and light framework

 Usage:
    One can use this class for code which needs to work in both the full and the
 light (i.e. FWLite) frameworks.  Data can be accessed using the same getByLabel
 interface which is available in the full framework.

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Aug 27 11:01:06 CDT 2009
//

// user include files
#include "DataFormats/Common/interface/BasicHandle.h"

#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Common/interface/ConvertHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Common/interface/TriggerResultsByName.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

// system include files
#include <typeinfo>

namespace edm {

  class ProcessHistory;
  class ProductID;
  class TriggerResults;
  class TriggerNames;
  class ParameterSet;

  class EventBase {
  public:
    EventBase();
    virtual ~EventBase();

    // ---------- const member functions ---------------------
    template <typename T>
    bool getByLabel(InputTag const&, Handle<T>&) const;

    template <typename T>
    bool getByToken(edm::EDGetTokenT<T> const& token, edm::Handle<T>& result) const;

    template <typename T>
    bool get(ProductID const&, Handle<T>&) const;

    // AUX functions.
    edm::EventID id() const { return eventAuxiliary().id(); }
    edm::Timestamp time() const { return eventAuxiliary().time(); }
    edm::LuminosityBlockNumber_t luminosityBlock() const { return eventAuxiliary().luminosityBlock(); }
    bool isRealData() const { return eventAuxiliary().isRealData(); }
    edm::EventAuxiliary::ExperimentType experimentType() const { return eventAuxiliary().experimentType(); }
    int bunchCrossing() const { return eventAuxiliary().bunchCrossing(); }
    int orbitNumber() const { return eventAuxiliary().orbitNumber(); }
    virtual edm::EventAuxiliary const& eventAuxiliary() const = 0;

    virtual TriggerNames const& triggerNames(edm::TriggerResults const& triggerResults) const = 0;
    virtual TriggerResultsByName triggerResultsByName(edm::TriggerResults const& triggerResults) const = 0;
    virtual ProcessHistory const& processHistory() const = 0;

    virtual edm::ParameterSet const* parameterSet(edm::ParameterSetID const& psID) const = 0;

  protected:
    static TriggerNames const* triggerNames_(edm::TriggerResults const& triggerResults);

    static edm::ParameterSet const* parameterSetForID_(edm::ParameterSetID const& psID);

  private:
    //EventBase(EventBase const&); // allow default

    //EventBase const& operator=(EventBase const&); // allow default

    virtual BasicHandle getByLabelImpl(std::type_info const& iWrapperType,
                                       std::type_info const& iProductType,
                                       InputTag const& iTag) const = 0;
    virtual BasicHandle getByTokenImpl(std::type_info const& iProductType, EDGetToken) const = 0;
    virtual BasicHandle getImpl(std::type_info const& iProductType, ProductID const& iTag) const = 0;
    // ---------- member data --------------------------------
  };

  template <typename T>
  bool EventBase::getByLabel(InputTag const& tag, Handle<T>& result) const {
    result.clear();
    BasicHandle bh = this->getByLabelImpl(typeid(edm::Wrapper<T>), typeid(T), tag);
    result = convert_handle<T>(std::move(bh));
    if (result.failedToGet()) {
      return false;
    }
    return true;
  }

  template <typename T>
  bool EventBase::getByToken(edm::EDGetTokenT<T> const& token, edm::Handle<T>& result) const {
    result.clear();
    edm::BasicHandle bh = this->getByTokenImpl(typeid(T), token);
    result = edm::convert_handle<T>(std::move(bh));
    if (result.failedToGet()) {
      return false;
    }
    return true;
  }

  template <typename T>
  bool EventBase::get(ProductID const& pid, Handle<T>& result) const {
    result.clear();
    BasicHandle bh = this->getImpl(typeid(T), pid);
    result = convert_handle_check_type<T>(std::move(bh));
    if (result.failedToGet()) {
      return false;
    }
    return true;
  }

}  // namespace edm

#endif

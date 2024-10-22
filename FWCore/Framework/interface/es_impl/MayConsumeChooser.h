#ifndef FWCore_Framework_es_impl_MayConsumeChooser_h
#define FWCore_Framework_es_impl_MayConsumeChooser_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     MayConsumeChooser
//
/**\class MayConsumeChooser MayConsumeChooser.h "MayConsumeChooser.h"

 Description: Handles calling the user's function to choose which product to get

 Usage:
    This is used by the mayConsumes ability of an ESProducer.

*/
//
// Original Author:  Chris Jones
//         Created:  Thu, 19 Sep 2019 15:29:50 GMT
//

// system include files
#include <type_traits>

// user include files
#include "FWCore/Framework/interface/es_impl/MayConsumeChooserBase.h"
#include "FWCore/Framework/interface/DataKey.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

// forward declarations

namespace edm::eventsetup::impl {
  template <typename RECBASE, typename PRODUCT, typename RCD, typename FUNC, typename PTAG>
  class MayConsumeChooser : public MayConsumeChooserBase<RECBASE> {
  public:
    using Record_t = typename PTAG::Record;
    using GetType_t = typename PTAG::Type;

    MayConsumeChooser(FUNC&& iFunc) : func_(std::forward<FUNC>(iFunc)) {}

    // ---------- const member functions ---------------------
    ESResolverIndex makeChoice(RECBASE const& iRecord) const final {
      return func_(this->tagGetter(), iRecord.getTransientHandle(token_));
    }

    EventSetupRecordKey recordKey() const noexcept final { return EventSetupRecordKey::makeKey<RCD>(); }
    TypeTag productType() const noexcept final { return DataKey::makeTypeTag<PRODUCT>(); }

    // ---------- static member functions --------------------

    // ---------- member functions ---------------------------
    edm::ESGetToken<GetType_t, Record_t>& token() { return token_; }

  private:
    // ---------- member data --------------------------------
    edm::ESGetToken<GetType_t, Record_t> token_;
    FUNC func_;
  };
}  // namespace edm::eventsetup::impl

#endif

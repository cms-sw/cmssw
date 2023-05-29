#ifndef DataFormats_FWLite_EventBase_h
#define DataFormats_FWLite_EventBase_h
// -*- C++ -*-
//
// Package:     FWLite
// Class  :     EventBase
//
/**\class EventBase EventBase.h DataFormats/FWLite/interface/EventBase.h

   Description: <one line class summary>

   Usage:
   <usage>

*/
//
// Original Author:  Charles Plager
//         Created:  Tue May  8 15:01:20 EDT 2007
//
// system include files
#include <string>
#include <typeinfo>
//
// // user include files
#include "FWCore/Common/interface/EventBase.h"

#include "Rtypes.h"

namespace edm {
  class BasicHandle;
  class ProductID;
  class WrapperBase;
}  // namespace edm

namespace fwlite {
  class EventBase : public edm::EventBase {
  public:
    EventBase();

    ~EventBase() override;

    virtual bool getByLabel(std::type_info const&, char const*, char const*, char const*, void*) const = 0;

    virtual edm::WrapperBase const* getByProductID(edm::ProductID const&) const = 0;

    using edm::EventBase::getByLabel;

    virtual std::string const getBranchNameFor(std::type_info const&, char const*, char const*, char const*) const = 0;

    virtual bool atEnd() const = 0;

    virtual EventBase const& operator++() = 0;

    virtual EventBase const& toBegin() = 0;

    virtual Long64_t fileIndex() const { return -1; }
    virtual Long64_t secondaryFileIndex() const { return -1; }

  protected:
    template <typename T>
    static edm::EDGetTokenT<T> makeTokenUsing(unsigned int iIndex) {
      return edm::EDGetTokenT<T>(iIndex);
    }

  private:
    virtual bool getByTokenImp(edm::EDGetToken, edm::WrapperBase const*&) const = 0;
    edm::BasicHandle getByLabelImpl(std::type_info const&, std::type_info const&, const edm::InputTag&) const override;
    edm::BasicHandle getByTokenImpl(std::type_info const&, edm::EDGetToken) const override;

    edm::BasicHandle getImpl(std::type_info const&, edm::ProductID const&) const override;
  };
}  // namespace fwlite

#endif

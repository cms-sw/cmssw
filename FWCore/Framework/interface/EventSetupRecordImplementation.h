#ifndef FWCore_Framework_EventSetupRecordImplementation_h
#define FWCore_Framework_EventSetupRecordImplementation_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     EventSetupRecordImplementation
//
/**\class EventSetupRecordImplementation EventSetupRecordImplementation.h FWCore/Framework/interface/EventSetupRecordImplementation.h

 Description: Help class which implements the necessary virtual methods for a new Record class

 Usage:
    This class handles implementing the necessary 'meta data' methods for a Record. To use the class, a new Record type should
 inherit from EventSetupRecordImplementation and pass itself as the argument to the template parameter. For example, for a
 Record named FooRcd, you would declare it like

      class FooRcd : public edm::eventsetup::EventSetupRecordImplementation< FooRcd > {};
*/
//
// Author:      Chris Jones
// Created:     Fri Apr  1 16:50:49 EST 2005
//

// user include files

#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/data_default_record_trait.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

// system include files

// forward declarations
namespace edm {
  namespace eventsetup {
    struct ComponentDescription;

    template <typename T>
    class EventSetupRecordImplementation : public EventSetupRecord {
    public:
      //virtual ~EventSetupRecordImplementation();

      // ---------- const member functions ---------------------
      EventSetupRecordKey key() const override { return EventSetupRecordKey::makeKey<T>(); }

      template <typename PRODUCT>
      ESHandle<PRODUCT> getHandle(ESGetToken<PRODUCT, T> const& iToken) const {
        return getHandleImpl<ESHandle>(iToken);
      }

      template <typename PRODUCT>
      ESHandle<PRODUCT> getHandle(ESGetToken<PRODUCT, edm::DefaultRecord> const& iToken) const {
        static_assert(std::is_same_v<T, eventsetup::default_record_t<ESHandle<PRODUCT>>>,
                      "The Record being used to retrieve the product is not the default record for the product type");
        return getHandleImpl<ESHandle>(iToken);
      }

      template <typename PRODUCT>
      ESTransientHandle<PRODUCT> getTransientHandle(ESGetToken<PRODUCT, T> const& iToken) const {
        return getHandleImpl<ESTransientHandle>(iToken);
      }

      template <typename PRODUCT>
      ESTransientHandle<PRODUCT> getTransientHandle(ESGetToken<PRODUCT, edm::DefaultRecord> const& iToken) const {
        static_assert(std::is_same_v<T, eventsetup::default_record_t<ESTransientHandle<PRODUCT>>>,
                      "The Record being used to retrieve the product is not the default record for the product type");
        return getHandleImpl<ESTransientHandle>(iToken);
      }

      template <typename PRODUCT>
      PRODUCT const& get(ESGetToken<PRODUCT, T> const& iToken) const {
        return *getHandleImpl<ESHandle>(iToken);
      }
      template <typename PRODUCT>
      PRODUCT const& get(ESGetToken<PRODUCT, T>& iToken) const {
        return *getHandleImpl<ESHandle>(const_cast<const ESGetToken<PRODUCT, T>&>(iToken));
      }

      template <typename PRODUCT>
      PRODUCT const& get(ESGetToken<PRODUCT, edm::DefaultRecord> const& iToken) const {
        static_assert(std::is_same_v<T, eventsetup::default_record_t<ESHandle<PRODUCT>>>,
                      "The Record being used to retrieve the product is not the default record for the product type");
        return *getHandleImpl<ESHandle>(iToken);
      }
      template <typename PRODUCT>
      PRODUCT const& get(ESGetToken<PRODUCT, edm::DefaultRecord>& iToken) const {
        return get(const_cast<const ESGetToken<PRODUCT, edm::DefaultRecord>&>(iToken));
      }

      // ---------- static member functions --------------------
      static EventSetupRecordKey keyForClass() { return EventSetupRecordKey::makeKey<T>(); }

      // ---------- member functions ---------------------------

    protected:
      EventSetupRecordImplementation() {}

    private:
      // ---------- member data --------------------------------
    };
  }  // namespace eventsetup
}  // namespace edm

#endif

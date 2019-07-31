#ifndef FWCore_Framework_EventSetup_h
#define FWCore_Framework_EventSetup_h
// -*- C++ -*-
//
// Package:     Framework
// Class:      EventSetup
//
/**\class edm::EventSetup

 Description: Container for all Records dealing with non-RunState info

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Thu Mar 24 13:50:04 EST 2005
//

// system include files
#include <cassert>
#include <map>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

// user include files
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Framework/interface/EventSetupImpl.h"
#include "FWCore/Framework/interface/HCMethods.h"
#include "FWCore/Framework/interface/NoRecordException.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Framework/interface/data_default_record_trait.h"
#include "FWCore/Utilities/interface/Transition.h"
#include "FWCore/Utilities/interface/ESIndices.h"

// forward declarations

namespace edm {

  class ESInputTag;
  template <class T, class R>
  class ESGetToken;
  class PileUp;

  namespace eventsetup {
    class EventSetupProvider;
    class EventSetupRecord;
    class EventSetupRecordImpl;
  }  // namespace eventsetup

  class EventSetup {
    ///Needed until a better solution can be found
    friend class edm::PileUp;

  public:
    explicit EventSetup(EventSetupImpl const& iSetup,
                        unsigned int iTransitionID,
                        ESProxyIndex const* iGetTokenIndices,
                        bool iRequireToken)
        : m_setup{iSetup}, m_getTokenIndices{iGetTokenIndices}, m_id{iTransitionID}, m_requireToken(iRequireToken) {}
    EventSetup(EventSetup const&) = delete;
    EventSetup& operator=(EventSetup const&) = delete;

    /** returns the Record of type T.  If no such record available
          a eventsetup::NoRecordException<T> is thrown */
    template <typename T>
    T get() const {
      using namespace eventsetup;
      using namespace eventsetup::heterocontainer;
      //NOTE: this will catch the case where T does not inherit from EventSetupRecord
      //  HOWEVER the error message under gcc 3.x is awful
      static_assert(std::is_base_of_v<edm::eventsetup::EventSetupRecord, T>,
                    "Trying to get a class that is not a Record from EventSetup");

      auto const temp = m_setup.findImpl(makeKey<typename type_from_itemtype<eventsetup::EventSetupRecordKey, T>::Type,
                                                 eventsetup::EventSetupRecordKey>());
      if (nullptr == temp) {
        throw eventsetup::NoRecordException<T>(recordDoesExist(m_setup, eventsetup::EventSetupRecordKey::makeKey<T>()));
      }
      T returnValue;
      returnValue.setImpl(temp, m_id, m_getTokenIndices, &m_setup, m_requireToken);
      return returnValue;
    }

    /** returns the Record of type T.  If no such record available
       a null optional is returned */
    template <typename T>
    std::optional<T> tryToGet() const {
      using namespace eventsetup;
      using namespace eventsetup::heterocontainer;

      //NOTE: this will catch the case where T does not inherit from EventSetupRecord
      static_assert(std::is_base_of_v<edm::eventsetup::EventSetupRecord, T>,
                    "Trying to get a class that is not a Record from EventSetup");
      auto const temp = impl().findImpl(makeKey<typename type_from_itemtype<eventsetup::EventSetupRecordKey, T>::Type,
                                                eventsetup::EventSetupRecordKey>());
      if (temp != nullptr) {
        T rec;
        rec.setImpl(temp, m_id, m_getTokenIndices, &m_setup, m_requireToken);
        return rec;
      }
      return std::nullopt;
    }

    /** can directly access data if data_default_record_trait<> is defined for this data type **/
    template <typename T>
    bool getData(T& iHolder) const {
      return getData(std::string{}, iHolder);
    }

    template <typename T>
    bool getData(const std::string& iLabel, T& iHolder) const {
      auto const& rec = this->get<eventsetup::default_record_t<T>>();
      return rec.get(iLabel, iHolder);
    }

    template <typename T>
    bool getData(const ESInputTag& iTag, T& iHolder) const {
      auto const& rec = this->get<eventsetup::default_record_t<T>>();
      return rec.get(iTag, iHolder);
    }

    template <typename T, typename R>
    T const& getData(const ESGetToken<T, R>& iToken) const noexcept(false) {
      return this
          ->get<std::conditional_t<std::is_same_v<R, edm::DefaultRecord>, eventsetup::default_record_t<ESHandle<T>>, R>>()
          .get(iToken);
    }
    template <typename T, typename R>
    T const& getData(ESGetToken<T, R>& iToken) const noexcept(false) {
      return this->getData(const_cast<const ESGetToken<T, R>&>(iToken));
    }

    template <typename T, typename R>
    ESHandle<T> getHandle(const ESGetToken<T, R>& iToken) const {
      if constexpr (std::is_same_v<R, edm::DefaultRecord>) {
        auto const& rec = this->get<eventsetup::default_record_t<ESHandle<T>>>();
        return rec.getHandle(iToken);
      } else {
        auto const& rec = this->get<R>();
        return rec.getHandle(iToken);
      }
    }

    template <typename T, typename R>
    ESTransientHandle<T> getTransientHandle(const ESGetToken<T, R>& iToken) const {
      if constexpr (std::is_same_v<R, edm::DefaultRecord>) {
        auto const& rec = this->get<eventsetup::default_record_t<ESTransientHandle<T>>>();
        return rec.getTransientHandle(iToken);
      } else {
        auto const& rec = this->get<R>();
        return rec.getTransientHandle(iToken);
      }
    }

    std::optional<eventsetup::EventSetupRecordGeneric> find(const eventsetup::EventSetupRecordKey& iKey) const {
      return m_setup.find(iKey, m_id, m_getTokenIndices);
    }

    ///clears the oToFill vector and then fills it with the keys for all available records
    void fillAvailableRecordKeys(std::vector<eventsetup::EventSetupRecordKey>& oToFill) const {
      m_setup.fillAvailableRecordKeys(oToFill);
    }
    ///returns true if the Record is provided by a Source or a Producer
    /// a value of true does not mean this EventSetup object holds such a record
    bool recordIsProvidedByAModule(eventsetup::EventSetupRecordKey const& iKey) const {
      return m_setup.recordIsProvidedByAModule(iKey);
    }
    // ---------- static member functions --------------------

  private:
    edm::EventSetupImpl const& impl() const { return m_setup; }

    // ---------- member data --------------------------------
    edm::EventSetupImpl const& m_setup;
    ESProxyIndex const* m_getTokenIndices;
    unsigned int m_id;
    bool m_requireToken;
  };

  // Free functions to retrieve an object from the EventSetup.
  // Will throw an exception if the record or  object are not found.

  template <typename T, typename R = typename eventsetup::data_default_record_trait<typename T::value_type>::type>
  T const& get(EventSetup const& setup) {
    ESHandle<T> handle;
    // throw if the record is not available
    setup.get<R>().get(handle);
    // throw if the handle is not valid
    return *handle.product();
  }

  template <typename T,
            typename R = typename eventsetup::data_default_record_trait<typename T::value_type>::type,
            typename L>
  T const& get(EventSetup const& setup, L&& label) {
    ESHandle<T> handle;
    // throw if the record is not available
    setup.get<R>().get(std::forward(label), handle);
    // throw if the handle is not valid
    return *handle.product();
  }

}  // namespace edm

#endif  // FWCore_Framework_EventSetup_h

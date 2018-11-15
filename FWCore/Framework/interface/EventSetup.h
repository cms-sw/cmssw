#ifndef FWCore_Framework_EventSetup_h
#define FWCore_Framework_EventSetup_h
// -*- C++ -*-
//
// Package:     Framework
// Class:      EventSetup
//
/**\class EventSetup EventSetup.h FWCore/Framework/interface/EventSetup.h

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
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Framework/interface/HCMethods.h"
#include "FWCore/Framework/interface/NoRecordException.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Framework/interface/data_default_record_trait.h"
#include "FWCore/Utilities/interface/Transition.h"

// forward declarations

namespace edm {
   class ActivityRegistry;
   class ESInputTag;
   template <class T>
   class ESGetTokenT;

   namespace eventsetup {
      class EventSetupProvider;
      class EventSetupRecord;
      class EventSetupRecordImpl;
      class EventSetupKnownRecordsSupplier;
   }

  class EventSetup
  {
    ///Only EventSetupProvider allowed to create a EventSetup
    friend class eventsetup::EventSetupProvider;
    public:
      virtual ~EventSetup();

      EventSetup(EventSetup const&) = delete;
      EventSetup& operator=(EventSetup const&) = delete;

      // ---------- const member functions ---------------------
      /** returns the Record of type T.  If no such record available
          a eventsetup::NoRecordException<T> is thrown */
      template< typename T>
         T get() const {
           using namespace eventsetup;
           using namespace eventsetup::heterocontainer;
            //NOTE: this will catch the case where T does not inherit from EventSetupRecord
            //  HOWEVER the error message under gcc 3.x is awful
            static_assert(std::is_base_of_v<edm::eventsetup::EventSetupRecord, T>, "Trying to get a class that is not a Record from EventSetup");

           auto const temp = findImpl(makeKey<typename type_from_itemtype<eventsetup::EventSetupRecordKey,T>::Type,eventsetup::EventSetupRecordKey>());
           if(nullptr == temp) {
             throw eventsetup::NoRecordException<T>(recordDoesExist(*this, eventsetup::EventSetupRecordKey::makeKey<T>()));
           }
           T returnValue;
           returnValue.setImpl(temp);
           return returnValue;
         }

      /** returns the Record of type T.  If no such record available
       a null pointer is returned */
      template< typename T>
        std::optional<T> tryToGet() const {
           using namespace eventsetup;
           using namespace eventsetup::heterocontainer;

           //NOTE: this will catch the case where T does not inherit from EventSetupRecord
           static_assert(std::is_base_of_v<edm::eventsetup::EventSetupRecord, T>,"Trying to get a class that is not a Record from EventSetup");
           auto const temp = findImpl(makeKey<typename type_from_itemtype<eventsetup::EventSetupRecordKey,T>::Type,eventsetup::EventSetupRecordKey>());
           if(temp != nullptr) {
              T rec;
              rec.setImpl(temp);
              return rec;
           }
           return std::nullopt;
        }

      /** can directly access data if data_default_record_trait<> is defined for this data type **/
      template< typename T>
         bool getData(T& iHolder) const {
            return getData(std::string{}, iHolder);
         }

      template< typename T>
         bool getData(const std::string& iLabel, T& iHolder) const {
            auto const& rec = this->get<eventsetup::default_record_t<T>>();
            return rec.get(iLabel, iHolder);
         }

      template< typename T>
        bool getData(const ESInputTag& iTag, T& iHolder) const {
           auto const& rec = this->get<eventsetup::default_record_t<T>>();
           return rec.get(iTag, iHolder);
        }

      template <typename T>
      bool getData(const ESGetTokenT<T>& iToken, ESHandle<T>& iHolder) const {
        return getData(iToken.m_tag, iHolder);
      }

      std::optional<eventsetup::EventSetupRecordGeneric> find(const eventsetup::EventSetupRecordKey&) const;

      ///clears the oToFill vector and then fills it with the keys for all available records
      void fillAvailableRecordKeys(std::vector<eventsetup::EventSetupRecordKey>& oToFill) const;

      ///returns true if the Record is provided by a Source or a Producer
      /// a value of true does not mean this EventSetup object holds such a record
      bool recordIsProvidedByAModule( eventsetup::EventSetupRecordKey const& ) const;
      // ---------- static member functions --------------------

      friend class eventsetup::EventSetupRecordImpl;

    protected:
      //Only called by EventSetupProvider
      void setKnownRecordsSupplier(eventsetup::EventSetupKnownRecordsSupplier const* iSupplier) {
        knownRecords_ = iSupplier;
      }

      void add(const eventsetup::EventSetupRecordImpl& iRecord);

      void clear();

    private:
      EventSetup(ActivityRegistry const*);

      ActivityRegistry const* activityRegistry() const { return activityRegistry_; }
      eventsetup::EventSetupRecordImpl const* findImpl(const eventsetup::EventSetupRecordKey&) const;

      void insert(const eventsetup::EventSetupRecordKey&,
                  const eventsetup::EventSetupRecordImpl*);

      // ---------- member data --------------------------------

      //NOTE: the records are not owned
      std::map<eventsetup::EventSetupRecordKey, eventsetup::EventSetupRecordImpl const *> recordMap_;
      eventsetup::EventSetupKnownRecordsSupplier const* knownRecords_;
      ActivityRegistry const* activityRegistry_;
  };

  // Free functions to retrieve an object from the EventSetup.
  // Will throw an exception if the record or  object are not found.

  template <typename T, typename R = typename eventsetup::data_default_record_trait<typename T::value_type>::type>
  T const& get(EventSetup const& setup) {
    ESHandle<T> handle;
    // throw if the record is not available
    setup.get<R>().get(handle);
    // throw if the handle is not valid
    return * handle.product();
  }

  template <typename T, typename R = typename eventsetup::data_default_record_trait<typename T::value_type>::type, typename L>
  T const& get(EventSetup const& setup, L && label) {
    ESHandle<T> handle;
    // throw if the record is not available
    setup.get<R>().get(std::forward(label), handle);
    // throw if the handle is not valid
    return * handle.product();
  }

}

#endif // FWCore_Framework_EventSetup_h

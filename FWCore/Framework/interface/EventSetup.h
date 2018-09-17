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

// forward declarations

namespace edm {
   class ActivityRegistry;
   class ESInputTag;

   namespace eventsetup {
      class EventSetupProvider;
      class EventSetupRecord;
      class EventSetupRecordImpl;
      template<class T> struct data_default_record_trait;
      class EventSetupKnownRecordsSupplier;
   }

  class EventSetup
  {
    ///Only EventSetupProvider allowed to create a EventSetup
    friend class eventsetup::EventSetupProvider;
    public:
      virtual ~EventSetup();

      // ---------- const member functions ---------------------
      /** returns the Record of type T.  If no such record available
          a eventsetup::NoRecordException<T> is thrown */
      template< typename T>
         T get() const {
           using namespace eventsetup;
           using namespace eventsetup::heterocontainer;
            //NOTE: this will catch the case where T does not inherit from EventSetupRecord
            //  HOWEVER the error message under gcc 3.x is awful
            static_assert(std::is_base_of<edm::eventsetup::EventSetupRecord, T>::value, "Trying to get a class that is not a Record from EventSetup");

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
           static_assert((std::is_base_of<edm::eventsetup::EventSetupRecord, T>::value),"Trying to get a class that is not a Record from EventSetup");
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
         void getData(T& iHolder) const {
            typedef typename T::value_type data_type;
            typedef typename eventsetup::data_default_record_trait<data_type>::type RecordT;
            const RecordT& rec = this->get<RecordT>();
            rec.get(iHolder);
         }
      template< typename T>
         void getData(const std::string& iLabel, T& iHolder) const {
            typedef typename T::value_type data_type;
            typedef typename eventsetup::data_default_record_trait<data_type>::type RecordT;
            const RecordT& rec = this->get<RecordT>();
            rec.get(iLabel,iHolder);
         }

      template< typename T>
        void getData(const edm::ESInputTag& iTag, T& iHolder) const {
           typedef typename T::value_type data_type;
           typedef typename eventsetup::data_default_record_trait<data_type>::type RecordT;
           const RecordT& rec = this->get<RecordT>();
           rec.get(iTag,iHolder);
        }

      std::optional<eventsetup::EventSetupRecordGeneric> find(const eventsetup::EventSetupRecordKey&) const;

      ///clears the oToFill vector and then fills it with the keys for all available records
      void fillAvailableRecordKeys(std::vector<eventsetup::EventSetupRecordKey>& oToFill) const;

      ///returns true if the Record is provided by a Source or a Producer
      /// a value of true does not mean this EventSetup object holds such a record
      bool recordIsProvidedByAModule( eventsetup::EventSetupRecordKey const& ) const;
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      template< typename T>
         void
         getAvoidCompilerBug(const T*& iValue) const {
            iValue = &(get<T>());
         }

      friend class eventsetup::EventSetupRecordImpl;

    protected:
      //Only called by EventSetupProvider
      void setKnownRecordsSupplier(eventsetup::EventSetupKnownRecordsSupplier const* iSupplier) {
        knownRecords_ = iSupplier;
      }

      void add(const eventsetup::EventSetupRecordImpl& iRecord);

      void clear();

    private:
      EventSetup(ActivityRegistry*);

      EventSetup(EventSetup const&) = delete; // stop default

      EventSetup const& operator=(EventSetup const&) = delete; // stop default

      ActivityRegistry* activityRegistry() const { return activityRegistry_; }
      eventsetup::EventSetupRecordImpl const* findImpl(const eventsetup::EventSetupRecordKey&) const;


      void insert(const eventsetup::EventSetupRecordKey&,
                  const eventsetup::EventSetupRecordImpl*);

      // ---------- member data --------------------------------

      //NOTE: the records are not owned
      std::map<eventsetup::EventSetupRecordKey, eventsetup::EventSetupRecordImpl const *> recordMap_;
      eventsetup::EventSetupKnownRecordsSupplier const* knownRecords_;
      ActivityRegistry* activityRegistry_;
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

#ifndef FWCore_Framework_EventSetupImpl_h
#define FWCore_Framework_EventSetupImpl_h
// -*- C++ -*-
//
// Package:     Framework
// Class:      EventSetupImpl
//
/**\class EventSetupImpl EventSetupImpl.h FWCore/Framework/interface/EventSetupImpl.h

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
   }

  class EventSetupImpl
  {
    ///Only EventSetupProvider allowed to create a EventSetup
    friend class eventsetup::EventSetupProvider;
    public:
      ~EventSetupImpl();

      EventSetupImpl(EventSetupImpl const&) = delete;
      EventSetupImpl& operator=(EventSetupImpl const&) = delete;

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

      void add(const eventsetup::EventSetupRecordImpl& iRecord);

      void clear();

    private:
      EventSetupImpl(ActivityRegistry const*);

      ActivityRegistry const* activityRegistry() const { return activityRegistry_; }
      eventsetup::EventSetupRecordImpl const* findImpl(const eventsetup::EventSetupRecordKey&) const;

      void insert(const eventsetup::EventSetupRecordKey&,
                  const eventsetup::EventSetupRecordImpl*);

      void setKeyIters(std::vector<eventsetup::EventSetupRecordKey>::const_iterator const& keysBegin,
                       std::vector<eventsetup::EventSetupRecordKey>::const_iterator const& keysEnd);

      // ---------- member data --------------------------------

      std::vector<eventsetup::EventSetupRecordKey>::const_iterator keysBegin_;
      std::vector<eventsetup::EventSetupRecordKey>::const_iterator keysEnd_;
      std::vector<eventsetup::EventSetupRecordImpl const*> recordImpls_;
      ActivityRegistry const* activityRegistry_;
  };

}

#endif // FWCore_Framework_EventSetup_h

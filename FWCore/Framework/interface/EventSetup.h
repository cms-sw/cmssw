#ifndef EVENTSETUP_EVENTSETUP_H
#define EVENTSETUP_EVENTSETUP_H
// -*- C++ -*-
//
// Package:     CoreFramework
// Class:      EventSetup
// 
/**\class EventSetup EventSetup.h Core/CoreFramework/interface/EventSetup.h

 Description: Container for all Records dealing with non-RunState info

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Thu Mar 24 13:50:04 EST 2005
//

// system include files
#include <map>
#include "boost/static_assert.hpp"
#include "boost/type_traits/is_base_and_derived.hpp"
// user include files
#include "FWCore/CoreFramework/interface/Timestamp.h"
#include "FWCore/CoreFramework/interface/EventSetupRecordKey.h"
#include "FWCore/CoreFramework/interface/HCMethods.h"

#include "FWCore/CoreFramework/interface/eventSetupGetImplementation.h"
// forward declarations

namespace edm {
   namespace eventsetup {
      class EventSetupProvider;
      class EventSetupRecord;
      template<class T> struct data_default_record_trait;
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
         const T& get() const {
            //NOTE: this will catch the case where T does not inherit from EventSetupRecord
            //  HOWEVER the error message under gcc 3.x is awful
            BOOST_STATIC_ASSERT((boost::is_base_and_derived<edm::eventsetup::EventSetupRecord, T>::value));
            const T* value = 0;
            eventSetupGetImplementation(*this, value);
            //NOTE: by construction, eventSetupGetImplementation should thrown an exception rather than return a null value
            assert(0 != value);
            return *value;
         }
      /** can directly access data if data_default_record_trait<> is defined for this data type **/
      template< typename T>
         void getData(T& iHolder) const {
            typedef typename T::value_type data_type;
            typedef typename eventsetup::data_default_record_trait<data_type>::type RecordT;
            const RecordT& rec = this->get<RecordT>();
            rec.get(iHolder);
         }
      
      const Timestamp& timestamp() const { return timestamp_;}

      const eventsetup::EventSetupRecord* find(const eventsetup::EventSetupRecordKey&) const;
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      template< typename T>
         void
         getAvoidCompilerBug(const T*& iValue) const {
            iValue = &(get<T>());
         }
   protected:
      //Only called by EventSetupProvider
      void setTimestamp(const Timestamp&);

      template<typename T>
         void add(const T& iRecord) {
            insert(eventsetup::heterocontainer::makeKey<T, eventsetup::EventSetupRecordKey>(), &iRecord);
         }
      
      void clear();
      
   private:
      EventSetup();
      
      EventSetup(EventSetup const&); // stop default

      EventSetup const& operator=(EventSetup const&); // stop default

      void insert(const eventsetup::EventSetupRecordKey&,
                  const eventsetup::EventSetupRecord*);

      // ---------- member data --------------------------------
      Timestamp timestamp_;
      
      //NOTE: the records are not owned
      std::map<eventsetup::EventSetupRecordKey, eventsetup::EventSetupRecord const *> recordMap_;
};

}
#endif /* EVENTSETUP_EVENTSETUP_H */

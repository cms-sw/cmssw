#ifndef EVENTSETUP_EVENTSETUPPROVIDER_H
#define EVENTSETUP_EVENTSETUPPROVIDER_H
// -*- C++ -*-
//
// Package:     Framework
// Class:      EventSetupProvider
// 
/**\class EventSetupProvider EventSetupProvider.h FWCore/Framework/interface/EventSetupProvider.h

 Description: Factory for a EventSetup

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Thu Mar 24 14:10:07 EST 2005
//

// system include files
#include <memory>
#include <map>

#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/EventSetup.h"

// forward declarations
namespace edm {
   class Timestamp;
   
   namespace eventsetup {
      class EventSetupRecordProvider;
      class DataProxyProvider;
      class EventSetupRecordIntervalFinder;
      
class EventSetupProvider
{

   public:
      EventSetupProvider();
      virtual ~EventSetupProvider();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      EventSetup const& eventSetupForInstance(const Timestamp&);
     
      //called by specializations of EventSetupRecordProviders
      template<class T>
         void addRecordToEventSetup(T& iRecord) {
            iRecord.setEventSetup(&eventSetup_);
            eventSetup_.add(iRecord);
         }
      
      void add(boost::shared_ptr<DataProxyProvider>);
      void add(boost::shared_ptr<EventSetupRecordIntervalFinder>);
      
      void finishConfiguration();
   protected:

      template <class T>
         void insert(std::auto_ptr<T> iRecordProvider) {
            std::auto_ptr<EventSetupRecordProvider> temp(iRecordProvider.release());
            insert(eventsetup::heterocontainer::makeKey<
                    typename T::RecordType, 
                       eventsetup::EventSetupRecordKey>(),
                    temp);
         }
      
   private:
      EventSetupProvider(const EventSetupProvider&); // stop default

      const EventSetupProvider& operator=(const EventSetupProvider&); // stop default


      void insert(const EventSetupRecordKey&, std::auto_ptr<EventSetupRecordProvider>);
      
      // ---------- member data --------------------------------
      EventSetup eventSetup_;
      typedef std::map<EventSetupRecordKey, boost::shared_ptr<EventSetupRecordProvider> > Providers;
      Providers providers_;
      bool mustFinishConfiguration_;
};

   }
}
#endif /* EVENTSETUP_EVENTSETUPPROVIDER_H */

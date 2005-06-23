#ifndef EVENTSETUP_DEPENDENTRECORDINTERVALFINDER_H
#define EVENTSETUP_DEPENDENTRECORDINTERVALFINDER_H
// -*- C++ -*-
//
// Package:     CoreFramework
// Class  :     DependentRecordIntervalFinder
// 
/**\class DependentRecordIntervalFinder DependentRecordIntervalFinder.h Core/CoreFramework/interface/DependentRecordIntervalFinder.h

 Description: Finds the intersection of the ValidityInterval for several Providers

 Usage:
    This class is used internally to a EventSetupRecordProvider which delivers a Record that is dependent on other Records.

    If no Providers are given, then Finder will always report an invalid ValidityInterval for all Timestamps

*/
//
// Author:      Chris Jones
// Created:     Sat Apr 30 19:36:59 EDT 2005
// $Id: DependentRecordIntervalFinder.h,v 1.1 2005/05/29 02:29:53 wmtan Exp $
//

// system include files
#include <vector>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/CoreFramework/interface/EventSetupRecordIntervalFinder.h"

// forward declarations
namespace edm {
   namespace eventsetup {
      class EventSetupRecordProvider;
      
class DependentRecordIntervalFinder : public EventSetupRecordIntervalFinder
{

   public:
      DependentRecordIntervalFinder(const EventSetupRecordKey&);
      virtual ~DependentRecordIntervalFinder();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      void addProviderWeAreDependentOn(boost::shared_ptr<EventSetupRecordProvider>);
      
   protected:
      virtual void setIntervalFor(const EventSetupRecordKey&,
                                   const Timestamp& , 
                                   ValidityInterval&);
      
   private:
      DependentRecordIntervalFinder(const DependentRecordIntervalFinder&); // stop default

      const DependentRecordIntervalFinder& operator=(const DependentRecordIntervalFinder&); // stop default

      // ---------- member data --------------------------------
      typedef std::vector< boost::shared_ptr<EventSetupRecordProvider> > Providers;
      Providers providers_;
};

   }
}
#endif /* EVENTSETUP_DEPENDENTRECORDINTERVALFINDER_H */

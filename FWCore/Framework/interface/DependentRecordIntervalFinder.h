#ifndef Framework_DependentRecordIntervalFinder_h
#define Framework_DependentRecordIntervalFinder_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     DependentRecordIntervalFinder
// 
/**\class DependentRecordIntervalFinder DependentRecordIntervalFinder.h FWCore/Framework/interface/DependentRecordIntervalFinder.h

 Description: Finds the intersection of the ValidityInterval for several Providers

 Usage:
    This class is used internally to a EventSetupRecordProvider which delivers a Record that is dependent on other Records.

    If no Providers are given, then Finder will always report an invalid ValidityInterval for all IOVSyncValues

*/
//
// Author:      Chris Jones
// Created:     Sat Apr 30 19:36:59 EDT 2005
// $Id: DependentRecordIntervalFinder.h,v 1.10 2010/12/17 04:31:53 chrjones Exp $
//

// system include files
#include <vector>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

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
      bool haveProviders() const {
        return !providers_.empty();
      }

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      void addProviderWeAreDependentOn(boost::shared_ptr<EventSetupRecordProvider>);
      
      void setAlternateFinder(boost::shared_ptr<EventSetupRecordIntervalFinder>);
   protected:
      virtual void setIntervalFor(const EventSetupRecordKey&,
                                   const IOVSyncValue& , 
                                   ValidityInterval&);
      
   private:
      DependentRecordIntervalFinder(const DependentRecordIntervalFinder&); // stop default

      const DependentRecordIntervalFinder& operator=(const DependentRecordIntervalFinder&); // stop default

      // ---------- member data --------------------------------
      typedef std::vector< boost::shared_ptr<EventSetupRecordProvider> > Providers;
      Providers providers_;
      
      boost::shared_ptr<EventSetupRecordIntervalFinder> alternate_;
      std::vector<ValidityInterval> previousIOVs_;
};

   }
}
#endif

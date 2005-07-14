#ifndef EVENTSETUP_DEPENDENTEVENTSETUPRECORDPROVIDER_H
#define EVENTSETUP_DEPENDENTEVENTSETUPRECORDPROVIDER_H
// -*- C++ -*-
//
// Package:     Framework
// Class  :     DependentEventSetupRecordProvider
// 
/**\class DependentEventSetupRecordProvider DependentEventSetupRecordProvider.h FWCore/Framework/interface/DependentEventSetupRecordProvider.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Sun May  1 16:30:01 EDT 2005
// $Id: DependentEventSetupRecordProvider.h,v 1.3 2005/06/23 19:59:30 wmtan Exp $
//

// system include files
#include <vector>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/EventSetupRecordProvider.h"

// forward declarations
namespace edm {
   namespace eventsetup {
      class DependentRecordIntervalFinder;
      
class DependentEventSetupRecordProvider : public EventSetupRecordProvider
{

   public:
      DependentEventSetupRecordProvider(const EventSetupRecordKey& iKey):
      EventSetupRecordProvider(iKey) {}
      //virtual ~DependentEventSetupRecordProvider();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

      void setDependentProviders(const std::vector< boost::shared_ptr<EventSetupRecordProvider> >&);

   private:
      DependentEventSetupRecordProvider(const DependentEventSetupRecordProvider&); // stop default

      const DependentEventSetupRecordProvider& operator=(const DependentEventSetupRecordProvider&); // stop default

      // ---------- member data --------------------------------
};

   }
}

#endif /* EVENTSETUP_DEPENDENTEVENTSETUPRECORDPROVIDER_H */

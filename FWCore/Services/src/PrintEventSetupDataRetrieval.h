#ifndef FWCore_Services_PrintEventSetupDataRetrieval_h
#define FWCore_Services_PrintEventSetupDataRetrieval_h
// -*- C++ -*-
//
// Package:     Services
// Class  :     PrintEventSetupDataRetrieval
// 
/**\class PrintEventSetupDataRetrieval PrintEventSetupDataRetrieval.h FWCore/Services/interface/PrintEventSetupDataRetrieval.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Jul  9 14:43:09 CDT 2009
//

// system include files
#include <map>

// user include files
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/DataKey.h"

// forward declarations
namespace edm {
  class ConfigurationDescriptions;

   class PrintEventSetupDataRetrieval {
      
   public:
      PrintEventSetupDataRetrieval(const ParameterSet&,ActivityRegistry&);

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

      // ---------- member functions ---------------------------
      void postProcessEvent(Event const&, EventSetup const&);
      void postBeginLumi(LuminosityBlock const&, EventSetup const&);
      void postBeginRun(Run const&, EventSetup const&);
      
   private:
      PrintEventSetupDataRetrieval(const PrintEventSetupDataRetrieval&); // stop default

      const PrintEventSetupDataRetrieval& operator=(const PrintEventSetupDataRetrieval&); // stop default

      void check(EventSetup const&);
      // ---------- member data --------------------------------
      typedef std::map<eventsetup::EventSetupRecordKey, std::pair<unsigned long long, std::map<eventsetup::DataKey,bool> > > RetrievedDataMap;
      RetrievedDataMap m_retrievedDataMap;
      std::vector<eventsetup::EventSetupRecordKey> m_recordKeys;
      bool m_printProviders;
};

   }

#endif

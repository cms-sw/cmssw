#ifndef FWCore_Framework_EventSetupsController_h
#define FWCore_Framework_EventSetupsController_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     EventSetupsController
// 
/**\class EventSetupsController EventSetupsController.h FWCore/Framework/interface/EventSetupsController.h

 Description: Manages a group of EventSetups which can share components

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Wed Jan 12 14:30:42 CST 2011
//

// user include files
#include "DataFormats/Provenance/interface/ParameterSetID.h"

// system include files
#include <boost/shared_ptr.hpp>

#include <map>
#include <vector>

// forward declarations
namespace edm {
  class EventSetupRecordIntervalFinder;
   class ParameterSet;
   class IOVSyncValue;
   
   namespace eventsetup {
      class EventSetupProvider;
      
      class EventSetupsController {
         
      public:
         EventSetupsController();
         //virtual ~EventSetupsController();
         
         // ---------- const member functions ---------------------
         
         // ---------- static member functions --------------------
         
         // ---------- member functions ---------------------------
         boost::shared_ptr<EventSetupProvider> makeProvider(ParameterSet&);

         void eventSetupForInstance(IOVSyncValue const& syncValue) const;

         void forceCacheClear() const;

         boost::shared_ptr<EventSetupRecordIntervalFinder> const* getAlreadyMadeESSource(ParameterSet const& pset) const;
         void putESSource(ParameterSet const& pset, boost::shared_ptr<EventSetupRecordIntervalFinder> const& component);
         void clearComponents();

      private:
         EventSetupsController(EventSetupsController const&); // stop default
         
         EventSetupsController const& operator=(EventSetupsController const&); // stop default
         
         // ---------- member data --------------------------------
         std::vector<boost::shared_ptr<EventSetupProvider> > providers_;

         std::multimap<ParameterSetID, std::pair<ParameterSet const*, boost::shared_ptr<EventSetupRecordIntervalFinder> > > essources_;
      };
   }
}
#endif

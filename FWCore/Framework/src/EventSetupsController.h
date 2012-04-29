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

// system include files
#include <boost/shared_ptr.hpp>

#include <vector>

// forward declarations
namespace edm {
   class ParameterSet;
   
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
         
      private:
         EventSetupsController(EventSetupsController const&); // stop default
         
         EventSetupsController const& operator=(EventSetupsController const&); // stop default
         
         // ---------- member data --------------------------------
         std::vector<boost::shared_ptr<EventSetupProvider> > providers_;
      };
   }
}
#endif

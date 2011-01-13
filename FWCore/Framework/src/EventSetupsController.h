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
// $Id$
//

// system include files
#include <vector>
#include <boost/shared_ptr.hpp>

// user include files

// forward declarations
namespace edm {
   class ParameterSet;
   class CommonParams;
   
   namespace eventsetup {
      class EventSetupProvider;
      
      class EventSetupsController {
         
      public:
         EventSetupsController();
         //virtual ~EventSetupsController();
         
         // ---------- const member functions ---------------------
         
         // ---------- static member functions --------------------
         
         // ---------- member functions ---------------------------
         boost::shared_ptr<EventSetupProvider> makeProvider(ParameterSet&, const CommonParams& );
         
      private:
         EventSetupsController(const EventSetupsController&); // stop default
         
         const EventSetupsController& operator=(const EventSetupsController&); // stop default
         
         // ---------- member data --------------------------------
         std::vector<boost::shared_ptr<EventSetupProvider> > providers_;
      };
   }
}


#endif

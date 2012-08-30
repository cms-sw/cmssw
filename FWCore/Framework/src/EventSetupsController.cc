// -*- C++ -*-
//
// Package:     Framework
// Class  :     EventSetupsController
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Wed Jan 12 14:30:44 CST 2011
// $Id$
//

// system include files

// user include files
#include "FWCore/Framework/src/EventSetupsController.h"
#include "FWCore/Framework/interface/EventSetupProviderMaker.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"

//
// constants, enums and typedefs
//

namespace edm {
   namespace eventsetup {
//
// static data member definitions
//

//
// constructors and destructor
//
EventSetupsController::EventSetupsController()
{
}

// EventSetupsController::EventSetupsController(const EventSetupsController& rhs)
// {
//    // do actual copying here;
// }

//EventSetupsController::~EventSetupsController()
//{
//}

//
// assignment operators
//
// const EventSetupsController& EventSetupsController::operator=(const EventSetupsController& rhs)
// {
//   //An exception safe implementation is
//   EventSetupsController temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
boost::shared_ptr<EventSetupProvider> 
EventSetupsController::makeProvider(ParameterSet& iPSet, const CommonParams& iParams)
{
   boost::shared_ptr<EventSetupProvider> returnValue(makeEventSetupProvider(iPSet) );

   fillEventSetupProvider(*returnValue, iPSet, iParams);
   
   providers_.push_back(returnValue);
   
   return returnValue;
}

//
// const member functions
//

//
// static member functions
//
   }
}

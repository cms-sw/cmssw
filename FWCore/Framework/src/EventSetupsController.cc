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
// $Id: EventSetupsController.cc,v 1.2 2012/03/27 19:52:30 wdd Exp $
//

// system include files

// user include files
#include "FWCore/Framework/src/EventSetupsController.h"
#include "FWCore/Framework/interface/EventSetupProviderMaker.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <algorithm>

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
EventSetupsController::makeProvider(ParameterSet& iPSet)
{
   boost::shared_ptr<EventSetupProvider> returnValue(makeEventSetupProvider(iPSet) );

   fillEventSetupProvider(*this, *returnValue, iPSet);
   
   providers_.push_back(returnValue);
   
   return returnValue;
}

void
EventSetupsController::eventSetupForInstance(IOVSyncValue const& syncValue) const {
  std::for_each(providers_.begin(), providers_.end(), [&syncValue](boost::shared_ptr<EventSetupProvider> const& esp) {
    esp->eventSetupForInstance(syncValue);
  });
}

void
EventSetupsController::forceCacheClear() const {
  std::for_each(providers_.begin(), providers_.end(), [](boost::shared_ptr<EventSetupProvider> const& esp) {
    esp->forceCacheClear();
  });
}

boost::shared_ptr<EventSetupRecordIntervalFinder> const*
EventSetupsController::getAlreadyMadeESSource(ParameterSet const& pset) const {
  auto elements = essources_.equal_range(pset.id());
  for (auto it = elements.first; it != elements.second; ++it) {
    if (isTransientEqual(pset, *it->second.first)) {
      return &it->second.second;
    }
  }
  return 0;
}

void
EventSetupsController::putESSource(ParameterSet const& pset, boost::shared_ptr<EventSetupRecordIntervalFinder> const& component) {
  essources_.insert(std::pair<ParameterSetID, std::pair<ParameterSet const*, boost::shared_ptr<EventSetupRecordIntervalFinder> > >(pset.id(), 
                                              std::pair<ParameterSet const*, boost::shared_ptr<EventSetupRecordIntervalFinder> >(&pset, component)));
}

void
EventSetupsController::clearComponents() {
  essources_.clear();
}

//
// const member functions
//

//
// static member functions
//
   }
}

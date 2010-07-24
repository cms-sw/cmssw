// -*- C++ -*-
//
// Package:     Python
// Class  :     EventWrapper
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Wed Jun 28 11:21:52 CDT 2006
//

// system include files

// user include files
#include "FWCore/Python/src/EventWrapper.h"

#include "FWCore/Framework/interface/Event.h"

//
// constants, enums and typedefs
//
using namespace edm::python;
//
// static data member definitions
//

//
// constructors and destructor
//
ConstEventWrapper::ConstEventWrapper(const edm::Event& iEvent):
event_(&iEvent)
{
}

// EventWrapper::EventWrapper(const EventWrapper& rhs)
// {
//    // do actual copying here;
// }

//EventWrapper::~EventWrapper()
//{
//}

//
// assignment operators
//
// const EventWrapper& EventWrapper::operator=(const EventWrapper& rhs)
// {
//   //An exception safe implementation is
//   EventWrapper temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
ConstEventWrapper::getByLabel(std::string const& iLabel, edm::GenericHandle& oHandle) const
{
  if(event_) {
    event_->getByLabel(iLabel,oHandle);
  }
}

void 
ConstEventWrapper::getByLabel(std::string const& iLabel, std::string const& iInstance, edm::GenericHandle& oHandle) const
{
  if(event_) {
    event_->getByLabel(iLabel,iInstance,oHandle);
  }
}
//
// const member functions
//

//
// static member functions
//

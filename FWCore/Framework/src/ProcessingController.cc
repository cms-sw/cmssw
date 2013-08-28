// -*- C++ -*-
//
// Package:     Framework
// Class  :     ProcessingController
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Mon Aug  9 09:33:31 CDT 2010
//

// system include files

// user include files
#include "FWCore/Framework/interface/ProcessingController.h"

using namespace edm;
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
ProcessingController::ProcessingController(ForwardState forwardState, ReverseState reverseState, bool iCanRandomAccess) :
forwardState_(forwardState),
reverseState_(reverseState),
transition_(kToNextEvent),
specifiedEvent_(),
canRandomAccess_(iCanRandomAccess),
lastOperationSucceeded_(true)
{
}

// ProcessingController::ProcessingController(const ProcessingController& rhs)
// {
//    // do actual copying here;
// }

//ProcessingController::~ProcessingController()
//{
//}

//
// assignment operators
//
// const ProcessingController& ProcessingController::operator=(const ProcessingController& rhs)
// {
//   //An exception safe implementation is
//   ProcessingController temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
ProcessingController::setTransitionToNextEvent()
{
   transition_ = kToNextEvent;
}

void 
ProcessingController::setTransitionToPreviousEvent()
{
   transition_ = kToPreviousEvent;
}

void 
ProcessingController::setTransitionToEvent( edm::EventID const& iID)
{
   transition_ = kToSpecifiedEvent;
   specifiedEvent_ =iID;
}

void
ProcessingController::setLastOperationSucceeded(bool value)
{
  lastOperationSucceeded_ = value;
}

//
// const member functions
//
ProcessingController::ForwardState 
ProcessingController::forwardState() const
{
   return forwardState_;
}

ProcessingController::ReverseState 
ProcessingController::reverseState() const
{
   return reverseState_;
}

bool 
ProcessingController::canRandomAccess() const
{
   return canRandomAccess_;
}

ProcessingController::Transition 
ProcessingController::requestedTransition() const
{
   return transition_;
}


edm::EventID 
ProcessingController::specifiedEventTransition() const
{
   return specifiedEvent_;
}

bool
ProcessingController::lastOperationSucceeded() const
{
  return lastOperationSucceeded_;
}

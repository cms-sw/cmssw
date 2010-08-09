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
// $Id$
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
ProcessingController::ProcessingController(ProcessingController::State iState, bool iCanRandomAccess):
state_(iState),
transition_(kToNextEvent),
specifiedEvent_(),
canRandomAccess_(iCanRandomAccess)
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

//
// const member functions
//
ProcessingController::State 
ProcessingController::processingState() const
{
   return state_;
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

// ---------- static member functions --------------------

// ---------- member functions ---------------------------


//
// static member functions
//

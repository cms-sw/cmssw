// -*- C++ -*-
//
// Package:     Modules
// Class  :     IterateNTimesLooper
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Jul 11 11:16:14 EDT 2006
// $Id: IterateNTimesLooper.cc,v 1.2 2006/07/28 13:24:35 valya Exp $
//

// system include files

// user include files
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Modules/src/IterateNTimesLooper.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
IterateNTimesLooper::IterateNTimesLooper(const edm::ParameterSet& iConfig) :
max_(iConfig.getParameter<unsigned int>("nTimes")),
times_(0),
shouldStop_(false)
{
}

// IterateNTimesLooper::IterateNTimesLooper(const IterateNTimesLooper& rhs)
// {
//    // do actual copying here;
// }

IterateNTimesLooper::~IterateNTimesLooper()
{
}

//
// assignment operators
//
// const IterateNTimesLooper& IterateNTimesLooper::operator=(const IterateNTimesLooper& rhs)
// {
//   //An exception safe implementation is
//   IterateNTimesLooper temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
IterateNTimesLooper::startingNewLoop(unsigned int iIteration) {
  times_ = iIteration;
  if (iIteration >= max_ ) {
    shouldStop_ = true;
  }
}

edm::EDLooper::Status 
IterateNTimesLooper::duringLoop(const edm::Event& event, const edm::EventSetup& eventSetup) {
  return shouldStop_ ? kStop : kContinue;
}

edm::EDLooper::Status 
IterateNTimesLooper::endOfLoop(const edm::EventSetup& es, unsigned int iCounter) {
  ++times_;
  return (times_ < max_ ) ? kContinue : kStop;
}

//
// const member functions
//

//
// static member functions
//

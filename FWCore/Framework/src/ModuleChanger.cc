// -*- C++ -*-
//
// Package:     Framework
// Class  :     ModuleChanger
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Thu Jul 15 15:05:10 EDT 2010
//

// system include files

// user include files
#include "FWCore/Framework/interface/ModuleChanger.h"
#include "FWCore/Framework/interface/Schedule.h"

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
ModuleChanger::ModuleChanger(Schedule* iSchedule):
schedule_(iSchedule)
{
}

// ModuleChanger::ModuleChanger(const ModuleChanger& rhs)
// {
//    // do actual copying here;
// }

ModuleChanger::~ModuleChanger()
{
}

//
// assignment operators
//
// const ModuleChanger& ModuleChanger::operator=(const ModuleChanger& rhs)
// {
//   //An exception safe implementation is
//   ModuleChanger temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

//
// const member functions
//
bool 
ModuleChanger::changeModule(const std::string& iLabel,
                            const ParameterSet& iPSet) const
{
   return schedule_->changeModule(iLabel,iPSet);
}

//
// static member functions
//

// -*- C++ -*-
//
// Package:     Framework
// Class  :     ScheduleInfo
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Thu Jul 15 19:40:14 CEST 2010
// $Id$
//

// system include files
#include <algorithm>
#include <iterator>
#include <boost/bind.hpp>
#include <functional>

// user include files
#include "FWCore/Framework/interface/ScheduleInfo.h"
#include "FWCore/Framework/interface/Schedule.h"

#include "FWCore/ParameterSet/interface/Registry.h"

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
ScheduleInfo::ScheduleInfo(const Schedule* iSchedule):
schedule_(iSchedule)
{
}

// ScheduleInfo::ScheduleInfo(const ScheduleInfo& rhs)
// {
//    // do actual copying here;
// }

ScheduleInfo::~ScheduleInfo()
{
}

//
// assignment operators
//
// const ScheduleInfo& ScheduleInfo::operator=(const ScheduleInfo& rhs)
// {
//   //An exception safe implementation is
//   ScheduleInfo temp(rhs);
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
void 
ScheduleInfo::availableModuleLabels(std::vector<std::string>& oLabelsToFill) const
{
   std::vector<ModuleDescription const*> desc = schedule_->getAllModuleDescriptions();
   
   oLabelsToFill.reserve(oLabelsToFill.size()+desc.size());
   std::transform(desc.begin(),desc.end(),
                  std::back_inserter(oLabelsToFill),
                  boost::bind(&ModuleDescription::moduleLabel,_1));
}

const ParameterSet*
ScheduleInfo::parametersForModule(const std::string& iLabel) const 
{
   std::vector<ModuleDescription const*> desc = schedule_->getAllModuleDescriptions();
   
   std::vector<ModuleDescription const*>::iterator itFound = std::find_if(desc.begin(),
                                                                          desc.end(),
                                                                          boost::bind(std::equal_to<std::string>(),
                                                                                      iLabel,
                                                                                      boost::bind(&ModuleDescription::moduleLabel,_1)));
   if (itFound == desc.end()) {
      return 0;
   }
   return pset::Registry::instance()->getMapped((*itFound)->parameterSetID());
}

void 
ScheduleInfo::availablePaths(std::vector<std::string>& oLabelsToFill) const
{
   schedule_->availablePaths(oLabelsToFill);
}

void 
ScheduleInfo::modulesInPath(const std::string& iPathLabel,
                            std::vector<std::string>& oLabelsToFill) const
{
   schedule_->modulesInPath(iPathLabel, oLabelsToFill);
}

//
// static member functions
//

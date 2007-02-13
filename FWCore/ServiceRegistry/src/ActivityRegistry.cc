// -*- C++ -*-
//
// Package:     ServiceRegistry
// Class  :     ActivityRegistry
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Sep  6 10:26:49 EDT 2005
// $Id: ActivityRegistry.cc,v 1.9 2006/08/08 00:35:59 chrjones Exp $
//

// system include files
#include <algorithm>

// user include files
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
//edm::ActivityRegistry::ActivityRegistry()
//{
//}
   
// ActivityRegistry::ActivityRegistry(const ActivityRegistry& rhs)
// {
//    // do actual copying here;
// }

//edm::ActivityRegistry::~ActivityRegistry()
//{
//}
   
//
// assignment operators
//
// const ActivityRegistry& ActivityRegistry::operator=(const ActivityRegistry& rhs)
// {
//   //An exception safe implementation is
//   ActivityRegistry temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
edm::ActivityRegistry::connect(ActivityRegistry& iOther)
{
   postBeginJobSignal_.connect(iOther.postBeginJobSignal_);
   postEndJobSignal_.connect(iOther.postEndJobSignal_);

   jobFailureSignal_.connect(iOther.jobFailureSignal_);

   preSourceSignal_.connect(iOther.preSourceSignal_);
   postSourceSignal_.connect(iOther.postSourceSignal_);
   
   preProcessEventSignal_.connect(iOther.preProcessEventSignal_);
   postProcessEventSignal_.connect(iOther.postProcessEventSignal_);

   preProcessPathSignal_.connect(iOther.preProcessPathSignal_);
   postProcessPathSignal_.connect(iOther.postProcessPathSignal_);

   preModuleSignal_.connect(iOther.preModuleSignal_);
   postModuleSignal_.connect(iOther.postModuleSignal_);

   preModuleConstructionSignal_.connect(iOther.preModuleConstructionSignal_);
   postModuleConstructionSignal_.connect(iOther.postModuleConstructionSignal_);

   preSourceConstructionSignal_.connect(iOther.preSourceConstructionSignal_);
   postSourceConstructionSignal_.connect(iOther.postSourceConstructionSignal_);

}

template<class T>
static
inline
void
copySlotsToFrom(T& iTo, T& iFrom)
{
  typename T::slot_list_type slots = iFrom.slots();
  
  std::for_each(slots.begin(),slots.end(),
                boost::bind( &T::connect, iTo, _1) );
}
void 
edm::ActivityRegistry::copySlotsFrom(ActivityRegistry& iOther)
{
  copySlotsToFrom(postBeginJobSignal_,iOther.postBeginJobSignal_);
  copySlotsToFrom(postEndJobSignal_,iOther.postEndJobSignal_);
  
  copySlotsToFrom(jobFailureSignal_,iOther.jobFailureSignal_);
  
  copySlotsToFrom(preSourceSignal_,iOther.preSourceSignal_);
  copySlotsToFrom(postSourceSignal_,iOther.postSourceSignal_);
  
  copySlotsToFrom(preProcessEventSignal_,iOther.preProcessEventSignal_);
  copySlotsToFrom(postProcessEventSignal_,iOther.postProcessEventSignal_);
  
  copySlotsToFrom(preProcessPathSignal_,iOther.preProcessPathSignal_);
  copySlotsToFrom(postProcessPathSignal_,iOther.postProcessPathSignal_);

  copySlotsToFrom(preModuleSignal_,iOther.preModuleSignal_);
  copySlotsToFrom(postModuleSignal_,iOther.postModuleSignal_);
  
  copySlotsToFrom(preModuleConstructionSignal_,iOther.preModuleConstructionSignal_);
  copySlotsToFrom(postModuleConstructionSignal_,iOther.postModuleConstructionSignal_);
  
  copySlotsToFrom(preSourceConstructionSignal_,iOther.preSourceConstructionSignal_);
  copySlotsToFrom(postSourceConstructionSignal_,iOther.postSourceConstructionSignal_);
  
}

//
// const member functions
//

//
// static member functions
//

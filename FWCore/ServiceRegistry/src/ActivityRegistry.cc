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
// $Id: ActivityRegistry.cc,v 1.11 2007/01/09 17:26:56 chrjones Exp $
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

   preModuleBeginJobSignal_.connect(iOther.preModuleBeginJobSignal_);
   postModuleBeginJobSignal_.connect(iOther.postModuleBeginJobSignal_);
   
   preModuleEndJobSignal_.connect(iOther.preModuleEndJobSignal_);
   postModuleEndJobSignal_.connect(iOther.postModuleEndJobSignal_);
   
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

template<class T>
static
inline
void
copySlotsToFromReverse(T& iTo, T& iFrom)
{
  // This handles service slots that are supposed to be in reverse
  // order of construction. Copying new ones in is a little
  // tricky.  Here is an example of what follows
  // slots in iTo before  4 3 2 1  and copy in slots in iFrom 8 7 6 5
  // reverse both  1 2 3 4  plus 5 6 7 8
  // then do the copy 1 2 3 4 5 6 7 8
  // then reverse back again to get the desired order
  // 8 7 6 5 4 3 2 1

  typename T::slot_list_type slotsFrom = iFrom.slots();
  typename T::slot_list_type slotsTo   = iTo.slots();
  
  std::reverse(slotsTo.begin(), slotsTo.end());
  std::reverse(slotsFrom.begin(), slotsFrom.end());

  std::for_each(slotsFrom.begin(),slotsFrom.end(),
                boost::bind( &T::connect, iTo, _1) );

  std::reverse(slotsTo.begin(), slotsTo.end());

  // Be nice and put these back in the state they were
  // at the beginning
  std::reverse(slotsFrom.begin(), slotsFrom.end());
}

void 
edm::ActivityRegistry::copySlotsFrom(ActivityRegistry& iOther)
{
  copySlotsToFrom(postBeginJobSignal_,iOther.postBeginJobSignal_);
  copySlotsToFromReverse(postEndJobSignal_,iOther.postEndJobSignal_);
  
  copySlotsToFromReverse(jobFailureSignal_,iOther.jobFailureSignal_);
  
  copySlotsToFrom(preSourceSignal_,iOther.preSourceSignal_);
  copySlotsToFromReverse(postSourceSignal_,iOther.postSourceSignal_);
  
  copySlotsToFrom(preProcessEventSignal_,iOther.preProcessEventSignal_);
  copySlotsToFromReverse(postProcessEventSignal_,iOther.postProcessEventSignal_);
  
  copySlotsToFrom(preProcessPathSignal_,iOther.preProcessPathSignal_);
  copySlotsToFromReverse(postProcessPathSignal_,iOther.postProcessPathSignal_);

  copySlotsToFrom(preModuleSignal_,iOther.preModuleSignal_);
  copySlotsToFromReverse(postModuleSignal_,iOther.postModuleSignal_);
  
  copySlotsToFrom(preModuleConstructionSignal_,iOther.preModuleConstructionSignal_);
  copySlotsToFromReverse(postModuleConstructionSignal_,iOther.postModuleConstructionSignal_);
  
  copySlotsToFrom(preModuleBeginJobSignal_,iOther.preModuleBeginJobSignal_);
  copySlotsToFromReverse(postModuleBeginJobSignal_,iOther.postModuleBeginJobSignal_);
  
  copySlotsToFrom(preModuleEndJobSignal_,iOther.preModuleEndJobSignal_);
  copySlotsToFromReverse(postModuleEndJobSignal_,iOther.postModuleEndJobSignal_);
  
  copySlotsToFrom(preSourceConstructionSignal_,iOther.preSourceConstructionSignal_);
  copySlotsToFromReverse(postSourceConstructionSignal_,iOther.postSourceConstructionSignal_);
  
}

//
// const member functions
//

//
// static member functions
//

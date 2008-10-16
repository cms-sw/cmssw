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
// $Id: ActivityRegistry.cc,v 1.13 2007/11/07 05:06:41 wmtan Exp $
//

// system include files
#include <algorithm>

// user include files
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Utilities/interface/Algorithms.h"

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
   
// ActivityRegistry::ActivityRegistry(ActivityRegistry const& rhs)
// {
//    // do actual copying here;
// }

//edm::ActivityRegistry::~ActivityRegistry()
//{
//}
   
//
// assignment operators
//
// ActivityRegistry const& ActivityRegistry::operator=(ActivityRegistry const& rhs)
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
   preSourceLumiSignal_.connect(iOther.preSourceLumiSignal_);
   preSourceRunSignal_.connect(iOther.preSourceRunSignal_);
   preSourceFileSignal_.connect(iOther.preSourceFileSignal_);
   postSourceSignal_.connect(iOther.postSourceSignal_);
   postSourceLumiSignal_.connect(iOther.postSourceLumiSignal_);
   postSourceRunSignal_.connect(iOther.postSourceRunSignal_);
   postSourceFileSignal_.connect(iOther.postSourceFileSignal_);
   
   preProcessEventSignal_.connect(iOther.preProcessEventSignal_);
   postProcessEventSignal_.connect(iOther.postProcessEventSignal_);

   preBeginRunSignal_.connect(iOther.preBeginRunSignal_);
   postBeginRunSignal_.connect(iOther.postBeginRunSignal_);

   preEndRunSignal_.connect(iOther.preEndRunSignal_);
   postEndRunSignal_.connect(iOther.postEndRunSignal_);

   preBeginLumiSignal_.connect(iOther.preBeginLumiSignal_);
   postBeginLumiSignal_.connect(iOther.postBeginLumiSignal_);

   preEndLumiSignal_.connect(iOther.preEndLumiSignal_);
   postEndLumiSignal_.connect(iOther.postEndLumiSignal_);

   preProcessPathSignal_.connect(iOther.preProcessPathSignal_);
   postProcessPathSignal_.connect(iOther.postProcessPathSignal_);

   prePathBeginRunSignal_.connect(iOther.prePathBeginRunSignal_);
   postPathBeginRunSignal_.connect(iOther.postPathBeginRunSignal_);

   prePathEndRunSignal_.connect(iOther.prePathEndRunSignal_);
   postPathEndRunSignal_.connect(iOther.postPathEndRunSignal_);

   prePathBeginLumiSignal_.connect(iOther.prePathBeginLumiSignal_);
   postPathBeginLumiSignal_.connect(iOther.postPathBeginLumiSignal_);

   prePathEndLumiSignal_.connect(iOther.prePathEndLumiSignal_);
   postPathEndLumiSignal_.connect(iOther.postPathEndLumiSignal_);

   preModuleSignal_.connect(iOther.preModuleSignal_);
   postModuleSignal_.connect(iOther.postModuleSignal_);

   preModuleBeginRunSignal_.connect(iOther.preModuleBeginRunSignal_);
   postModuleBeginRunSignal_.connect(iOther.postModuleBeginRunSignal_);

   preModuleEndRunSignal_.connect(iOther.preModuleEndRunSignal_);
   postModuleEndRunSignal_.connect(iOther.postModuleEndRunSignal_);

   preModuleBeginLumiSignal_.connect(iOther.preModuleBeginLumiSignal_);
   postModuleBeginLumiSignal_.connect(iOther.postModuleBeginLumiSignal_);

   preModuleEndLumiSignal_.connect(iOther.preModuleEndLumiSignal_);
   postModuleEndLumiSignal_.connect(iOther.postModuleEndLumiSignal_);

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
  
  edm::for_all(slots, boost::bind( &T::connect, iTo, _1) );
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

  edm::for_all(slotsFrom, boost::bind( &T::connect, iTo, _1) );

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
  copySlotsToFrom(preSourceLumiSignal_,iOther.preSourceLumiSignal_);
  copySlotsToFrom(preSourceRunSignal_,iOther.preSourceRunSignal_);
  copySlotsToFrom(preSourceFileSignal_,iOther.preSourceFileSignal_);
  copySlotsToFromReverse(postSourceSignal_,iOther.postSourceSignal_);
  copySlotsToFromReverse(postSourceLumiSignal_,iOther.postSourceLumiSignal_);
  copySlotsToFromReverse(postSourceRunSignal_,iOther.postSourceRunSignal_);
  copySlotsToFromReverse(postSourceFileSignal_,iOther.postSourceFileSignal_);
  
  copySlotsToFrom(preProcessEventSignal_,iOther.preProcessEventSignal_);
  copySlotsToFromReverse(postProcessEventSignal_,iOther.postProcessEventSignal_);
  
  copySlotsToFrom(preBeginRunSignal_,iOther.preBeginRunSignal_);
  copySlotsToFromReverse(postBeginRunSignal_,iOther.postBeginRunSignal_);

  copySlotsToFrom(preEndRunSignal_,iOther.preEndRunSignal_);
  copySlotsToFromReverse(postEndRunSignal_,iOther.postEndRunSignal_);

  copySlotsToFrom(preBeginLumiSignal_,iOther.preBeginLumiSignal_);
  copySlotsToFromReverse(postBeginLumiSignal_,iOther.postBeginLumiSignal_);

  copySlotsToFrom(preEndLumiSignal_,iOther.preEndLumiSignal_);
  copySlotsToFromReverse(postEndLumiSignal_,iOther.postEndLumiSignal_);

  copySlotsToFrom(preProcessPathSignal_,iOther.preProcessPathSignal_);
  copySlotsToFromReverse(postProcessPathSignal_,iOther.postProcessPathSignal_);

  copySlotsToFrom(prePathBeginRunSignal_,iOther.prePathBeginRunSignal_);
  copySlotsToFromReverse(postPathBeginRunSignal_,iOther.postPathBeginRunSignal_);

  copySlotsToFrom(prePathEndRunSignal_,iOther.prePathEndRunSignal_);
  copySlotsToFromReverse(postPathEndRunSignal_,iOther.postPathEndRunSignal_);

  copySlotsToFrom(prePathBeginLumiSignal_,iOther.prePathBeginLumiSignal_);
  copySlotsToFromReverse(postPathBeginLumiSignal_,iOther.postPathBeginLumiSignal_);

  copySlotsToFrom(prePathEndLumiSignal_,iOther.prePathEndLumiSignal_);
  copySlotsToFromReverse(postPathEndLumiSignal_,iOther.postPathEndLumiSignal_);

  copySlotsToFrom(preModuleSignal_,iOther.preModuleSignal_);
  copySlotsToFromReverse(postModuleSignal_,iOther.postModuleSignal_);
  
  copySlotsToFrom(preModuleBeginRunSignal_,iOther.preModuleBeginRunSignal_);
  copySlotsToFromReverse(postModuleBeginRunSignal_,iOther.postModuleBeginRunSignal_);
  
  copySlotsToFrom(preModuleEndRunSignal_,iOther.preModuleEndRunSignal_);
  copySlotsToFromReverse(postModuleEndRunSignal_,iOther.postModuleEndRunSignal_);
  
  copySlotsToFrom(preModuleBeginLumiSignal_,iOther.preModuleBeginLumiSignal_);
  copySlotsToFromReverse(postModuleBeginLumiSignal_,iOther.postModuleBeginLumiSignal_);
  
  copySlotsToFrom(preModuleEndLumiSignal_,iOther.preModuleEndLumiSignal_);
  copySlotsToFromReverse(postModuleEndLumiSignal_,iOther.postModuleEndLumiSignal_);
  
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

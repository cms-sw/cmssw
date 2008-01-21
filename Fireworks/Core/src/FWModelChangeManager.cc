// -*- C++ -*-
//
// Package:     Core
// Class  :     FWModelChangeManager
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Jan 17 19:13:46 EST 2008
// $Id$
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWModelChangeManager.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWModelChangeManager::FWModelChangeManager():
m_depth(0)
{
}

// FWModelChangeManager::FWModelChangeManager(const FWModelChangeManager& rhs)
// {
//    // do actual copying here;
// }

FWModelChangeManager::~FWModelChangeManager()
{
}

//
// assignment operators
//
// const FWModelChangeManager& FWModelChangeManager::operator=(const FWModelChangeManager& rhs)
// {
//   //An exception safe implementation is
//   FWModelChangeManager temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
FWModelChangeManager::beginChanges()
{++m_depth;}

void 
FWModelChangeManager::changed(const FWModelId& iID)
{  
   FWChangeSentry sentry(*this);
   m_changes.insert(iID);   
}

void 
FWModelChangeManager::endChanges()
{
   assert(m_depth !=0);
   if(0 == --m_depth) {
      if(not m_changes.empty()) {
         changes_(m_changes);
         m_changes.clear();
      }
   }
}

//
// const member functions
//

//
// static member functions
//

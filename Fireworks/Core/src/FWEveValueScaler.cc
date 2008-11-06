// -*- C++ -*-
//
// Package:     Core
// Class  :     FWEveValueScaler
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Jul  3 10:30:04 EDT 2008
// $Id: FWEveValueScaler.cc,v 1.2 2008/07/09 06:54:26 jmuelmen Exp $
//

// system include files
#include <assert.h>
#include "TEveElement.h"
#include "TEveManager.h"

// user include files
#include "Fireworks/Core/interface/FWEveValueScaler.h"
#include "Fireworks/Core/interface/FWEveValueScaled.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
/*
FWEveValueScaler::FWEveValueScaler()
{
}
*/
// FWEveValueScaler::FWEveValueScaler(const FWEveValueScaler& rhs)
// {
//    // do actual copying here;
// }

FWEveValueScaler::~FWEveValueScaler()
{
}

//
// assignment operators
//
// const FWEveValueScaler& FWEveValueScaler::operator=(const FWEveValueScaler& rhs)
// {
//   //An exception safe implementation is
//   FWEveValueScaler temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
FWEveValueScaler::addElement(TEveElement* iElement)
{
   FWEveValueScaled* scaled=dynamic_cast<FWEveValueScaled*> (iElement);
   assert(0!=scaled);
   m_scalables.AddElement(iElement);

   scaled->setScale(m_scale);
   iElement->ElementChanged(kTRUE,kTRUE);
}

void
FWEveValueScaler::removeElement(TEveElement* iElement)
{
   m_scalables.RemoveElement(iElement);
}
void
FWEveValueScaler::removeElements()
{
   m_scalables.RemoveElements();
}

void
FWEveValueScaler::setScale(float iScale)
{
   if(iScale != m_scale) {
      m_scale = iScale;
      TEveManager::TRedrawDisabler disableRedraw(gEve);

      for(TEveElement::List_i it = m_scalables.BeginChildren(), itEnd=m_scalables.EndChildren();
          it != itEnd;
          ++it) {
         FWEveValueScaled* scaled=dynamic_cast<FWEveValueScaled*> (*it);
         scaled->setScale(m_scale);
         (*it)->ElementChanged(kTRUE,kTRUE);
      }
   }
}

//
// const member functions
//

//
// static member functions
//

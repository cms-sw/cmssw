// -*- C++ -*-
//
// Package:     Core
// Class  :     FWCustomIconsButton
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Oct 23 13:05:35 EDT 2008
// $Id: FWCustomIconsButton.cc,v 1.8 2012/11/16 02:15:25 amraktad Exp $
//

// system include files
#include <algorithm>
#include <assert.h>
#include "TGPicture.h"

// user include files
#include "Fireworks/Core/interface/FWCustomIconsButton.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWCustomIconsButton::FWCustomIconsButton(const TGWindow* iParent,
                                         const TGPicture* iUpIcon,
                                         const TGPicture* iDownIcon,
                                         const TGPicture* iDisabledIcon,
                                         const TGPicture* iBelowMouseIcon,
                                         Int_t id, GContext_t norm, UInt_t option) :
   TGButton(iParent,id, norm, option),
   m_upIcon(iUpIcon),
   m_downIcon(iDownIcon),
   m_disabledIcon(iDisabledIcon),
   m_belowMouseIcon(iBelowMouseIcon),
   m_inside(false)
{
   assert(0!=iUpIcon);
   assert(0!=iDownIcon);
   assert(0!=iDisabledIcon);
   gVirtualX->ShapeCombineMask(GetId(), 0, 0, iUpIcon->GetMask());
   SetBackgroundPixmap(iUpIcon->GetPicture());
   Resize(iUpIcon->GetWidth(),iUpIcon->GetHeight());
   fTWidth = iUpIcon->GetWidth();
   fTHeight = iUpIcon->GetHeight();
}

// FWCustomIconsButton::FWCustomIconsButton(const FWCustomIconsButton& rhs)
// {
//    // do actual copying here;
// }

FWCustomIconsButton::~FWCustomIconsButton()
{
}

//
// assignment operators
//
// const FWCustomIconsButton& FWCustomIconsButton::operator=(const FWCustomIconsButton& rhs)
// {
//   //An exception safe implementation is
//   FWCustomIconsButton temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
FWCustomIconsButton::swapIcons(const TGPicture*& iUpIcon,
                               const TGPicture*& iDownIcon,
                               const TGPicture*& iDisabledIcon)
{
   std::swap(iUpIcon,m_upIcon);
   std::swap(iDownIcon,m_downIcon);
   std::swap(iDisabledIcon,m_disabledIcon);
   gVirtualX->ShapeCombineMask(GetId(), 0, 0, m_upIcon->GetMask());
   fClient->NeedRedraw(this);
}

void
FWCustomIconsButton::setIcons(const TGPicture* iUpIcon,
                              const TGPicture* iDownIcon,
                              const TGPicture* iDisabledIcon,
                              const TGPicture* iBelowMouseIcon)
{
   m_upIcon       = iUpIcon;
   m_downIcon     = iDownIcon;
   m_disabledIcon = iDisabledIcon;
   m_belowMouseIcon = iBelowMouseIcon;

   gVirtualX->ShapeCombineMask(GetId(), 0, 0, m_upIcon->GetMask());
   fClient->NeedRedraw(this);
}

//
// const member functions
//

//______________________________________________________________________________
bool FWCustomIconsButton::HandleCrossing(Event_t *event)
{
   if (event->fType == kEnterNotify)
      m_inside = true;      
   else if (event->fType == kLeaveNotify)   
      m_inside = false;

   fClient->NeedRedraw(this);

   return TGButton::HandleCrossing(event);
}


void
FWCustomIconsButton::DoRedraw()
{
   //ChangeOptions(0);
   //TGButton::DoRedraw();
   //Stole this from TGPictureButton.
   int x = (fWidth - fTWidth) >> 1;
   int y = (fHeight - fTHeight) >> 1;

   switch(fState)
   {
      case kButtonUp:
         if (m_belowMouseIcon && m_inside)
              m_belowMouseIcon->Draw(fId, fNormGC,x,y);
         else
              m_upIcon->Draw(fId, fNormGC,x,y);
         break;
      case kButtonEngaged:
      case kButtonDown:
         m_downIcon->Draw(fId, fNormGC,x,y);
         break;
      case kButtonDisabled:
      default:
         m_disabledIcon->Draw(fId, fNormGC,x,y);
   }
}

//
// static member functions
//

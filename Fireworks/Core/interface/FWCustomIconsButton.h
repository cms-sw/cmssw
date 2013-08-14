#ifndef Fireworks_Core_FWCustomIconsButton_h
#define Fireworks_Core_FWCustomIconsButton_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWCustomIconsButton
//
/**\class FWCustomIconsButton FWCustomIconsButton.h Fireworks/Core/interface/FWCustomIconsButton.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Thu Oct 23 13:05:30 EDT 2008
// $Id: FWCustomIconsButton.h,v 1.6 2009/11/29 15:56:33 amraktad Exp $
//

// system include files
#include "TGButton.h"

// user include files

// forward declarations
class TGPicture;

class FWCustomIconsButton : public TGButton
{

public:
   FWCustomIconsButton(const TGWindow*  iParent,
                       const TGPicture* iUpIcon,
                       const TGPicture* iDownIcon,
                       const TGPicture* iDisableIcon,
                       const TGPicture* iBelowMouseIcon = 0,
                       Int_t id = -1,
                       GContext_t norm = TGButton::GetDefaultGC() (),
                       UInt_t option=0);
   
   virtual ~FWCustomIconsButton();

   virtual bool HandleCrossing(Event_t*);
   
   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   void swapIcons(const TGPicture*& iUpIcon,
                  const TGPicture*& iDownIcon,
                  const TGPicture*& iDisabledIcon);

   void setIcons(const TGPicture* iUpIcon,
                 const TGPicture* iDownIcon,
                 const TGPicture* iDisabledIcon,
                 const TGPicture* ibelowMouseIcon = 0);

   const TGPicture* upIcon() const { return m_upIcon; }
   const TGPicture* downIcon() const { return m_downIcon; }
   const TGPicture* disabledIcon() const { return m_disabledIcon; }
   const TGPicture* bellowMouseIcon() const { return m_belowMouseIcon; }

protected:
   virtual void DoRedraw();
private:
   FWCustomIconsButton(const FWCustomIconsButton&); // stop default

   const FWCustomIconsButton& operator=(const FWCustomIconsButton&); // stop default

   // ---------- member data --------------------------------
   const TGPicture* m_upIcon;
   const TGPicture* m_downIcon;
   const TGPicture* m_disabledIcon;
   const TGPicture* m_belowMouseIcon;
   
   bool m_inside;   
};


#endif

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
// $Id: FWCustomIconsButton.h,v 1.1 2008/11/05 09:08:25 chrjones Exp $
//

// system include files
#include "TGButton.h"

// user include files

// forward declarations
class TGPicture;

class FWCustomIconsButton : public TGButton
{

public:
   FWCustomIconsButton(const TGWindow* iParent,
                       const TGPicture* iUpIcon,
                       const TGPicture* iDownIcon,
                       const TGPicture* iDisabledIcon,
                       Int_t id = -1,
                       GContext_t norm = TGButton::GetDefaultGC()(),
                       UInt_t option=0);
   virtual ~FWCustomIconsButton();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   void swapIcons(const TGPicture*& iUpIcon,
                  const TGPicture*& iDownIcon,
                  const TGPicture*& iDisabledIcon);

protected:
   virtual void DoRedraw();
private:
   FWCustomIconsButton(const FWCustomIconsButton&); // stop default

   const FWCustomIconsButton& operator=(const FWCustomIconsButton&); // stop default

   // ---------- member data --------------------------------
   const TGPicture* m_upIcon;
   const TGPicture* m_downIcon;
   const TGPicture* m_disabledIcon;

};


#endif

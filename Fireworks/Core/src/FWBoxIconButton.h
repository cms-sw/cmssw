#ifndef Fireworks_Core_FWBoxIconButton_h
#define Fireworks_Core_FWBoxIconButton_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWBoxIconButton
// 
/**\class FWBoxIconButton FWBoxIconButton.h Fireworks/Core/interface/FWBoxIconButton.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 19 19:04:08 CST 2009
//

// system include files
#include "TGButton.h"

// user include files

// forward declarations
class FWBoxIconBase;

class FWBoxIconButton : public TGButton {

public:
   FWBoxIconButton(const TGWindow* iParent,
                   FWBoxIconBase* iBase,
                   Int_t iID=-1,
                   GContext_t norm = TGButton::GetDefaultGC() (),
                   UInt_t option=0);
   ~FWBoxIconButton() override;
   
   // ---------- const member functions ---------------------
   
   // ---------- static member functions --------------------
   
   // ---------- member functions ---------------------------
   void setNormCG(GContext_t);
protected:
   void DoRedraw() override;
private:
   FWBoxIconButton(const FWBoxIconButton&) = delete; // stop default
   
   const FWBoxIconButton& operator=(const FWBoxIconButton&) = delete; // stop default
   
   // ---------- member data --------------------------------
   FWBoxIconBase* m_iconBase;
};


#endif

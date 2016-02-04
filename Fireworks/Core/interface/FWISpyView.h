#ifndef Fireworks_Core_FWISpyView_h
#define Fireworks_Core_FWISpyView_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWISpyView
// 
/**\class FWISpyView FWISpyView.h Fireworks/Core/interface/FWISpyView.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Wed Apr  7 14:41:32 CEST 2010
// $Id: FWISpyView.h,v 1.2 2010/04/16 13:44:06 amraktad Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FW3DViewBase.h"

// forward declarations

class FWISpyView : public FW3DViewBase
{

public:
   FWISpyView(TEveWindowSlot*, FWViewType::EType);
   virtual ~FWISpyView();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

private:
   FWISpyView(const FWISpyView&); // stop default

   const FWISpyView& operator=(const FWISpyView&); // stop default

   // ---------- member data --------------------------------

};


#endif

#ifndef Fireworks_Core_FW3DRecHitView_h
#define Fireworks_Core_FW3DRecHitView_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FW3DRecHitView
// 
/**\class FW3DRecHitView FW3DRecHitView.h Fireworks/Core/interface/FW3DRecHitView.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Wed Apr  7 14:41:32 CEST 2010
// $Id$
//

// system include files

// user include files
#include "Fireworks/Core/interface/FW3DViewBase.h"

// forward declarations

class FW3DRecHitView : public FW3DViewBase
{

public:
   FW3DRecHitView(TEveWindowSlot*, TEveScene*);
   virtual ~FW3DRecHitView();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

private:
   FW3DRecHitView(const FW3DRecHitView&); // stop default

   const FW3DRecHitView& operator=(const FW3DRecHitView&); // stop default

   // ---------- member data --------------------------------

};


#endif

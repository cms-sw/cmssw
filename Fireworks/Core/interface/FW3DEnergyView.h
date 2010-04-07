#ifndef Fireworks_Core_FW3DEnergyView_h
#define Fireworks_Core_FW3DEnergyView_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FW3DEnergyView
// 
/**\class FW3DEnergyView FW3DEnergyView.h Fireworks/Core/interface/FW3DEnergyView.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Wed Apr  7 14:41:26 CEST 2010
// $Id$
//

// system include files

// user include files
#include "Fireworks/Core/interface/FW3DViewBase.h"

// forward declarations

class FW3DEnergyView: public FW3DViewBase
{
public:
   FW3DEnergyView(TEveWindowSlot*, TEveScene*);
   virtual ~FW3DEnergyView();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

private:
   FW3DEnergyView(const FW3DEnergyView&); // stop default

   const FW3DEnergyView& operator=(const FW3DEnergyView&); // stop default

   // ---------- member data --------------------------------
};


#endif

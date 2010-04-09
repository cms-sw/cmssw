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
// $Id: FW3DEnergyView.h,v 1.1 2010/04/07 16:56:20 amraktad Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FW3DViewBase.h"

// forward declarations
class TEveCalo3D;

class FW3DEnergyView: public FW3DViewBase
{
public:
   FW3DEnergyView(TEveWindowSlot*, TEveScene*);
   virtual ~FW3DEnergyView();

   virtual void setGeometry(fireworks::Context&);
   
   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

private:
   FW3DEnergyView(const FW3DEnergyView&); // stop default

   const FW3DEnergyView& operator=(const FW3DEnergyView&); // stop default

   // ---------- member data --------------------------------
   TEveCalo3D* m_calo;
};


#endif

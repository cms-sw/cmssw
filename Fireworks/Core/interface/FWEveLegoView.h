#ifndef Fireworks_Core_FWEveLegoView_h
#define Fireworks_Core_FWEveLegoView_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWEveLegoView
// 
/**\class FWEveLegoView FWEveLegoView.h Fireworks/Core/interface/FWEveLegoView.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Mon May 31 13:09:38 CEST 2010
// $Id: FWEveLegoView.h,v 1.32 2010/11/21 11:18:13 amraktad Exp $
//

#include "Fireworks/Core/interface/FWLegoViewBase.h"

class TEveStraightLineSet;

class FWEveLegoView: public FWLegoViewBase
{
public:
   FWEveLegoView(TEveWindowSlot*, FWViewType::EType);
   virtual ~FWEveLegoView();

   virtual void setContext(const fireworks::Context&);
   virtual void setBackgroundColor(Color_t);

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

private:
   FWEveLegoView(const FWEveLegoView&); // stop default

   const FWEveLegoView& operator=(const FWEveLegoView&); // stop default

   // ---------- member data --------------------------------
   TEveStraightLineSet* m_boundaries;
};


#endif

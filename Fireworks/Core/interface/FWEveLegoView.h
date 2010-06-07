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
// $Id: FWEveLegoView.h,v 1.27 2010/05/31 13:01:24 amraktad Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWLegoViewBase.h"

// forward declarations

class FWEveLegoView: public FWLegoViewBase
{
public:
   FWEveLegoView(TEveWindowSlot*, FWViewType::EType);
   virtual ~FWEveLegoView();

   virtual void setContext(fireworks::Context&);

   // ---------- const member functions ---------------------

   virtual TEveCaloData* getCaloData(fireworks::Context&) const;

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

private:
   FWEveLegoView(const FWEveLegoView&); // stop default

   const FWEveLegoView& operator=(const FWEveLegoView&); // stop default

   // ---------- member data --------------------------------
};


#endif

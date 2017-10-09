#ifndef Fireworks_Core_FWHFView_h
#define Fireworks_Core_FWHFView_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWHFView
// 
/**\class FWHFView FWHFView.h Fireworks/Core/interface/FWHFView.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Yanjun
//         Created:  Mon May 31 13:42:21 CEST 2010
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWLegoViewBase.h"

// forward declarations

class FWHFView : public FWLegoViewBase
{
public:
   FWHFView(TEveWindowSlot*, FWViewType::EType);
   ~FWHFView() override;

   void setContext(const fireworks::Context&) override;
   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

private:
   FWHFView(const FWHFView&) = delete; // stop default

   const FWHFView& operator=(const FWHFView&) = delete; // stop default

   // ---------- member data --------------------------------
};


#endif

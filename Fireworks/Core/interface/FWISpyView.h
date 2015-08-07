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
//

// system include files

// user include files
#include "Fireworks/Core/interface/FW3DViewBase.h"

// forward declarations

class FWISpyView : public FW3DViewBase
{
public:
   FWISpyView(TEveWindowSlot*, FWViewType::EType, unsigned int version=9);
   virtual ~FWISpyView();
   virtual void setContext(const fireworks::Context& x);

   virtual void populateController(ViewerParameterGUI&) const;
   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

private:
   FWISpyView(const FWISpyView&); // stop default

   const FWISpyView& operator=(const FWISpyView&); // stop default

   // ---------- member data --------------------------------};
};

#endif

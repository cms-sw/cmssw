#ifndef Fireworks_Core_FW3DView_h
#define Fireworks_Core_FW3DView_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FW3DView
// 
/**\class FW3DView FW3DView.h Fireworks/Core/interface/FW3DView.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Wed Apr  7 14:41:26 CEST 2010
//

// system include files

// user include files
#include "Fireworks/Core/interface/FW3DViewBase.h"

// forward declarations
class TEveCalo3D;

class FW3DView: public FW3DViewBase
{
public:
   FW3DView(TEveWindowSlot*, FWViewType::EType);
   virtual ~FW3DView();

   virtual void setContext(const fireworks::Context&);
   virtual TEveCaloViz* getEveCalo() const;
   
   
   // ---------- const member functions ---------------------

   //   virtual void populateController(ViewerParameterGUI&) const;
   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

private:
   FW3DView(const FW3DView&); // stop default

   const FW3DView& operator=(const FW3DView&); // stop default
   
   // ---------- member data --------------------------------
   TEveCalo3D*        m_calo;

};


#endif

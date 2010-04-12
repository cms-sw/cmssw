#ifndef Fireworks_Core_FWEveLegoView_h
#define Fireworks_Core_FWEveLegoView_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWEveLegoView
//
/**\class FWEveLegoView FWEveLegoView.h Fireworks/Core/interface/FWEveLegoView.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 21 11:22:37 EST 2008
// $Id: FWEveLegoView.h,v 1.24 2010/04/09 17:23:57 amraktad Exp $
//

// system include files
#include "Rtypes.h"

// user include files
#include "Fireworks/Core/interface/FWEveView.h"
#include "Fireworks/Core/interface/FWBoolParameter.h"
#include "Fireworks/Core/interface/FWDoubleParameter.h"
#include "Fireworks/Core/interface/FWEvePtr.h"

// forward declarations
class TEveViewer;
class TEveScene;
class TEveElementList;
class TEveCaloLego;
class TEveCaloLegoOverlay;
class TGLMatrix;


class FWEveLegoView : public FWEveView
{
public:
   FWEveLegoView(TEveWindowSlot*, TEveScene*);
   virtual ~FWEveLegoView();

   virtual void setContext(fireworks::Context&);
   virtual void setFrom(const FWConfiguration&);

   // ---------- const member functions ---------------------

   virtual void addTo(FWConfiguration&) const;


   // ---------- member functions ---------------------------
   void finishSetup();
   void setMinEnergy();

private:
   FWEveLegoView(const FWEveLegoView&);    // stop default

   const FWEveLegoView& operator=(const FWEveLegoView&);    // stop default

   void setMinEcalEnergy(double);
   void setMinHcalEnergy(double);
   void setCameras();
   void setAutoRebin();
   void setPixelsPerBin();
   void plotEt();
   void showScales();
   void updateLegoScale();
   
   // ---------- member data --------------------------------
   TEveCaloLego*        m_lego;
   TEveCaloLegoOverlay* m_overlay;
   
   FWBoolParameter   m_plotEt;
   FWBoolParameter   m_autoRebin;
   FWDoubleParameter m_pixelsPerBin;
   FWBoolParameter   m_showScales;
   FWDoubleParameter m_legoFixedScale;
   FWBoolParameter   m_legoAutoScale;
};


#endif

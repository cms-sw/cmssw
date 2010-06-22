#ifndef Fireworks_Core_FWLegoViewBase_h
#define Fireworks_Core_FWLegoViewBase_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWLegoViewBase
//
/**\class FWLegoViewBase FWLegoViewBase.h Fireworks/Core/interface/FWLegoViewBase.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 21 11:22:37 EST 2008
// $Id: FWLegoViewBase.h,v 1.4 2010/06/18 19:51:24 amraktad Exp $
//

// system include files
#include "Rtypes.h"

// user include files
#include "Fireworks/Core/interface/FWEveView.h"
#include "Fireworks/Core/interface/FWBoolParameter.h"
#include "Fireworks/Core/interface/FWDoubleParameter.h"
#include "Fireworks/Core/interface/FWLongParameter.h"
#include "Fireworks/Core/interface/FWEvePtr.h"

// forward declarations
class TEveViewer;
class TEveScene;
class TEveElementList;
class TEveCaloLego;
class TEveCaloLegoOverlay;
class TEveCaloDataHist;

class FWLegoViewBase : public FWEveView
{
public:
   FWLegoViewBase(TEveWindowSlot*, FWViewType::EType);
   virtual ~FWLegoViewBase();

   virtual void setFrom(const FWConfiguration&);

   virtual void setContext(fireworks::Context&);

   virtual void eventBegin();
   virtual void eventEnd();
   // ---------- const member functions ---------------------

   virtual void addTo(FWConfiguration&) const;
   virtual TEveCaloData* getCaloData(fireworks::Context&) const = 0;

   // ---------- member functions ---------------------------
   void finishSetup();
   void setMinEnergy();

protected:
   TEveCaloLego*        m_lego;
   TEveCaloLegoOverlay* m_overlay;

private:
   FWLegoViewBase(const FWLegoViewBase&);    // stop default

   const FWLegoViewBase& operator=(const FWLegoViewBase&);    // stop default

   void setMinEcalEnergy(double);
   void setMinHcalEnergy(double);
   void setCameras();
   void setAutoRebin();
   void setPixelsPerBin();
   void setFontSizein2D();
   void plotEt();
   void autoScale();
   void updateLegoScale();
   void showScales();
   
   // ---------- member data --------------------------------
   
   FWBoolParameter   m_plotEt;
   FWBoolParameter   m_autoRebin;
   FWDoubleParameter m_pixelsPerBin;
   FWLongParameter   m_drawValuesIn2D;
   FWBoolParameter   m_showScales;
   FWDoubleParameter m_legoFixedScale;
   FWBoolParameter   m_legoAutoScale;
};


#endif

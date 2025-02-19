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
// $Id: FWLegoViewBase.h,v 1.14 2010/11/15 17:14:27 amraktad Exp $
//

// system include files
#include "Rtypes.h"

// user include files
#include "Fireworks/Core/interface/FWEveView.h"
#include "Fireworks/Core/interface/FWBoolParameter.h"
#include "Fireworks/Core/interface/FWDoubleParameter.h"
#include "Fireworks/Core/interface/FWLongParameter.h"
#include "Fireworks/Core/interface/Context.h"

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

   virtual void setContext(const fireworks::Context&);

   // ---------- const member functions ---------------------

   virtual void addTo(FWConfiguration&) const;
   virtual void populateController(ViewerParameterGUI&) const;

   virtual TEveCaloViz* getEveCalo() const;

   // ---------- member functions ---------------------------

protected:
   TEveCaloLego*        m_lego;
   TEveCaloLegoOverlay* m_overlay;

private:
   FWLegoViewBase(const FWLegoViewBase&);    // stop default

   const FWLegoViewBase& operator=(const FWLegoViewBase&);    // stop default

   void setCameras();
   void setAutoRebin();
   void setPixelsPerBin();
   void setFontSizein2D();
   void autoScale();
   void showOverlay();
   void setProjectionMode();
   void setCell2DMode();
   
   // ---------- member data --------------------------------
   
   FWBoolParameter   m_autoRebin;
   FWDoubleParameter m_pixelsPerBin;
   FWEnumParameter   m_projectionMode; 
   FWEnumParameter   m_cell2DMode;
   FWLongParameter   m_drawValuesIn2D;
   FWBoolParameter   m_showOverlay;
};


#endif

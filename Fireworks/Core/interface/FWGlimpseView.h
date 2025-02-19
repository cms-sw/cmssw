#ifndef Fireworks_Core_FWGlimpseView_h
#define Fireworks_Core_FWGlimpseView_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGlimpseView
//
/**\class FWGlimpseView FWGlimpseView.h Fireworks/Core/interface/FWGlimpseView.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 21 11:22:37 EST 2008
// $Id: FWGlimpseView.h,v 1.17 2010/09/02 18:10:10 amraktad Exp $
//

// system include files
#include "Rtypes.h"

// user include files
#include "Fireworks/Core/interface/FWEveView.h"
#include "Fireworks/Core/interface/FWBoolParameter.h"

// forward declarations
class TEveViewer;
class TEveScene;
class TEveElementList;
class TEveWindowSlot;
class TEveGeoShape;

class FWGlimpseView : public FWEveView
{
public:
   FWGlimpseView(TEveWindowSlot*, FWViewType::EType);
   virtual ~FWGlimpseView();

   // ---------- const member functions ---------------------

   virtual void addTo(FWConfiguration&) const;
   virtual void setFrom(const FWConfiguration&);

   // ---------- static member functions --------------------

private:
   FWGlimpseView(const FWGlimpseView&);    // stop default
   const FWGlimpseView& operator=(const FWGlimpseView&);    // stop default

   void createAxis();
   void showAxes( );
   void showCylinder( );

   // ---------- member data --------------------------------
   TEveGeoShape*  m_cylinder;

   // FWDoubleParameter m_scaleParam;
   FWBoolParameter   m_showAxes;
   FWBoolParameter   m_showCylinder;
};


#endif

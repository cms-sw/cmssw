#ifndef Fireworks_Core_FW3DView_h
#define Fireworks_Core_FW3DView_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FW3DView
//
/**\class FW3DView FW3DView.h Fireworks/Core/interface/FW3DView.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 21 11:22:37 EST 2008
// $Id: FW3DView.h,v 1.17 2010/03/25 16:29:19 matevz Exp $
//

// system include files

// user include files
#include "Rtypes.h"
#include "Fireworks/Core/interface/FWEveView.h"
#include "Fireworks/Core/interface/FWLongParameter.h"
#include "Fireworks/Core/interface/FWBoolParameter.h"

// forward declarations
class TEveElementList;
class TEveGeoShape;
class TEveWindowSlot;

class DetIdToMatrix;
class FW3DViewGeometry;
class FWColorManager;

class FW3DView : public FWEveView
{
public:
   FW3DView(TEveWindowSlot*, TEveScene*);
   virtual ~FW3DView();

   // ---------- const member functions ---------------------

   virtual void addTo(FWConfiguration&) const;
   virtual void setFrom(const FWConfiguration&);

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   virtual void setGeometry( const DetIdToMatrix* geom, FWColorManager&);

   // To be fixed.
   void updateGlobalSceneScaleParameters();

private:
   FW3DView(const FW3DView&);    // stop default

   const FW3DView& operator=(const FW3DView&);    // stop default

   // ---------- member data --------------------------------
   FW3DViewGeometry*  m_geometry;

   // parameters
   FWBoolParameter m_showMuonBarrel;
   FWBoolParameter m_showMuonEndcap;
   FWBoolParameter m_showPixelBarrel;
   FWBoolParameter m_showPixelEndcap;
   FWBoolParameter m_showTrackerBarrel;
   FWBoolParameter m_showTrackerEndcap;
   FWBoolParameter m_showWireFrame;
   FWLongParameter m_geomTransparency;
   FWDoubleParameter  m_caloFixedScale;
   FWBoolParameter    m_caloAutoScale;
   
   void showWireFrame( bool );
};


#endif

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
// $Id: FW3DView.h,v 1.16 2010/03/16 11:51:53 amraktad Exp $
//

// system include files
#include "Rtypes.h"

// user include files
#include "Fireworks/Core/interface/FWEveView.h"
#include "Fireworks/Core/interface/FWLongParameter.h"
#include "Fireworks/Core/interface/FWBoolParameter.h"

// forward declarations
class TGLMatrix;
class TEveElementList;
class TEveGeoShape;
class TEveWindowSlot;
class DetIdToMatrix;

class FW3DView : public FWEveView
{
public:
   FW3DView(TEveWindowSlot*, TEveElementList*);
   virtual ~FW3DView();

   // ---------- const member functions ---------------------
   const std::string& typeName() const;

   virtual void addTo(FWConfiguration&) const;
   virtual void setFrom(const FWConfiguration&);

   // ---------- static member functions --------------------
   static const std::string& staticTypeName();

   // ---------- member functions ---------------------------
   void setGeometry( const DetIdToMatrix* geom );

   // To be fixed.
   void updateGlobalSceneScaleParameters();

private:
   FW3DView(const FW3DView&);    // stop default

   const FW3DView& operator=(const FW3DView&);    // stop default
   void showMuonBarrel( );
   void showMuonEndcap( );
   void showPixelBarrel( );
   void showPixelEndcap( );
   void showTrackerBarrel( );
   void showTrackerEndcap( );
   void showWireFrame( );
   void setTransparency( );

   // ---------- member data --------------------------------
   TEveScene* m_detectorScene;

   TGLMatrix* m_cameraMatrix;
   TGLMatrix* m_cameraMatrixBase;
   Double_t*  m_cameraFOV;

   DetIdToMatrix*     m_geometry;

   TEveElementList*   m_muonBarrelElements;
   TEveElementList*   m_muonEndcapElements;
   TEveElementList*   m_pixelBarrelElements;
   TEveElementList*   m_pixelEndcapElements;
   TEveElementList*   m_trackerBarrelElements;
   TEveElementList*   m_trackerEndcapElements;

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
};


#endif

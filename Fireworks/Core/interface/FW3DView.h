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
// $Id: FW3DView.h,v 1.8 2009/01/23 21:35:40 amraktad Exp $
//

// system include files
#include "Rtypes.h"

// user include files
#include "Fireworks/Core/interface/FWViewBase.h"
#include "Fireworks/Core/interface/FWLongParameter.h"
#include "Fireworks/Core/interface/FWDoubleParameter.h"
#include "Fireworks/Core/interface/FWBoolParameter.h"

// forward declarations
class TGFrame;
class TGLEmbeddedViewer;
class TEvePad;
class TEveViewer;
class TEveScene;
class TEveElementList;
class TEveGeoShape;
class TGLMatrix;
class FW3DViewManager;
class DetIdToMatrix;
class TEveWindowSlot;

class FW3DView : public FWViewBase
{

public:
   FW3DView(TEveWindowSlot*, TEveElementList*);
   virtual ~FW3DView();

   // ---------- const member functions ---------------------
   TGFrame* frame() const;
   const std::string& typeName() const;
   virtual void addTo(FWConfiguration&) const;

   virtual void saveImageTo(const std::string& iName) const;

   // ---------- static member functions --------------------
   static const std::string& staticTypeName();

   // ---------- member functions ---------------------------
   virtual void setFrom(const FWConfiguration&);
   void makeGeometry( const DetIdToMatrix* geom );

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
   TEveViewer* m_viewer;
   TGLEmbeddedViewer* m_embeddedViewer;
   TEveScene* m_scene;
   TEveScene* m_detectorScene;

   TGLMatrix* m_cameraMatrix;
   TGLMatrix* m_cameraMatrixBase;
   Double_t*  m_cameraFOV;

   TEveElement*      m_muonBarrelElements;
   TEveElement*      m_muonEndcapElements;
   TEveElement*      m_pixelBarrelElements;
   TEveElement*      m_pixelEndcapElements;
   TEveElement*      m_trackerBarrelElements;
   TEveElement*      m_trackerEndcapElements;
   FWBoolParameter m_showMuonBarrel;
   FWBoolParameter m_showMuonEndcap;
   FWBoolParameter m_showPixelBarrel;
   FWBoolParameter m_showPixelEndcap;
   FWBoolParameter m_showTrackerBarrel;
   FWBoolParameter m_showTrackerEndcap;
   FWBoolParameter m_showWireFrame;

   FWLongParameter m_geomTransparency;
};


#endif

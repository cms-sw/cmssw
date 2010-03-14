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
// $Id: FW3DView.h,v 1.14 2010/03/08 12:34:26 amraktad Exp $
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
class FWEventAnnotation;
class CmsAnnotation;
class FWViewContextMenuHandlerGL;

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
   virtual void setFrom(const FWConfiguration&);
   virtual FWViewContextMenuHandlerBase* contextMenuHandler() const;

   // ---------- static member functions --------------------
   static const std::string& staticTypeName();

   // ---------- member functions ---------------------------
   void setGeometry( const DetIdToMatrix* geom );
   void setBackgroundColor(Color_t);
   void eventEnd();

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
   void lineWidthChanged();


   // ---------- member data --------------------------------
   TEveViewer* m_viewer;
   TGLEmbeddedViewer* m_embeddedViewer;
   TEveScene* m_scene;
   TEveScene* m_detectorScene;
   boost::shared_ptr<FWViewContextMenuHandlerGL>   m_viewContextMenu;
   FWEventAnnotation* m_overlayEventInfo; 
   CmsAnnotation*     m_overlayLogo;

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
   FWLongParameter m_overlayEventInfoLevel;
   FWBoolParameter    m_drawCMSLogo;
   FWBoolParameter m_showMuonBarrel;
   FWBoolParameter m_showMuonEndcap;
   FWBoolParameter m_showPixelBarrel;
   FWBoolParameter m_showPixelEndcap;
   FWBoolParameter m_showTrackerBarrel;
   FWBoolParameter m_showTrackerEndcap;
   FWBoolParameter m_showWireFrame;
   FWLongParameter m_geomTransparency;

   FWDoubleParameter m_lineWidth;
};


#endif

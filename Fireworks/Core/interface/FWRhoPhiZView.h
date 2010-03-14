#ifndef Fireworks_Core_FWRhoPhiZView_h
#define Fireworks_Core_FWRhoPhiZView_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWRhoPhiZView
//
/**\class FWRhoPhiZView FWRhoPhiZView.h Fireworks/Core/interface/FWRhoPhiZView.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Tue Feb 19 10:33:21 EST 2008
// $Id: FWRhoPhiZView.h,v 1.27 2010/03/08 12:34:52 amraktad Exp $
//

// system include files
#include <string>
#include "TGLViewer.h"
#include "TEveProjections.h"

// user include files
#include "Fireworks/Core/interface/FWViewBase.h"
#include "Fireworks/Core/interface/FWDoubleParameter.h"
#include "Fireworks/Core/interface/FWBoolParameter.h"
#include "Fireworks/Core/interface/FWEvePtr.h"

// forward declarations
class TEvePad;
class TEveViewer;
class TGLEmbeddedViewer;
class TEveProjectionManager;
class TGFrame;
class TGLMatrix;
class TEveCalo2D;
class TEveScene;
class TEveProjectionAxes;
class TEveWindowSlot;
class FWEventAnnotation;
class CmsAnnotation;
class FWRhoPhiZViewManager;
class FWViewContextMenuHandlerGL;


class FWRhoPhiZView : public FWViewBase
{

public:
   FWRhoPhiZView(TEveWindowSlot* iParent,
                 const std::string& iTypeName,
                 const TEveProjection::EPType_e& iProjType);
   virtual ~FWRhoPhiZView();

   // ---------- const member functions ---------------------
   TGFrame* frame() const;
   const std::string& typeName() const;

   virtual void addTo(FWConfiguration&) const;
   virtual void saveImageTo(const std::string& iName) const;
   virtual void setFrom(const FWConfiguration&);
   virtual FWViewContextMenuHandlerBase* contextMenuHandler() const;

   // ---------- member functions ---------------------------
   void resetCamera();
   void destroyElements();
   void replicateGeomElement(TEveElement*);
   void showProjectionAxes( );
   void setBackgroundColor(Color_t);
   void eventEnd();

   //returns the new element created from this import
   TEveElement* importElements(TEveElement* iProjectableChild, float iLayer, TEveElement* iProjectedParent=0);

private:
   void doDistortion();
   void doCompression(bool);
   void doZoom(double);
   void updateCaloParameters();
   void updateScaleParameters();
   void updateCalo(TEveElement*, bool dataChanged = false);
   void updateCaloLines(TEveElement*);
   // void setMinEnergy( TEveCalo2D* calo, double value, std::string name );
   void lineWidthChanged();
   void lineSmoothnessChanged();

   FWRhoPhiZView(const FWRhoPhiZView&);    // stop default

   const FWRhoPhiZView& operator=(const FWRhoPhiZView&);    // stop default

   // ---------- member data --------------------------------
   FWEvePtr<TEveViewer> m_viewer;
   TGLEmbeddedViewer* m_embeddedViewer;
   FWEvePtr<TEveProjectionManager> m_projMgr;
   TEveProjection::EPType_e m_projType;
   std::vector<TEveElement*> m_geom;
   std::string m_typeName;
   FWEvePtr<TEveScene> m_scene;
   double m_caloScale;
   FWEvePtr<TEveProjectionAxes> m_axes;
   boost::shared_ptr<FWViewContextMenuHandlerGL>   m_viewContextMenu;
   FWEventAnnotation* m_overlayEventInfo;   
   CmsAnnotation*     m_overlayLogo;

   // parameters
   FWLongParameter    m_overlayEventInfoLevel;
   FWBoolParameter    m_drawCMSLogo;
   FWDoubleParameter  m_caloDistortion;
   FWDoubleParameter  m_muonDistortion;
   FWBoolParameter    m_showProjectionAxes;
   FWBoolParameter    m_compressMuon;
   FWDoubleParameter  m_caloFixedScale;
   FWBoolParameter    m_caloAutoScale;
   FWDoubleParameter  m_lineWidth;
   FWBoolParameter    m_smoothLine;
   FWBoolParameter*   m_showHF;
   FWBoolParameter*   m_showEndcaps;

   // camera parameters
   double* m_cameraZoom;
   TGLMatrix* m_cameraMatrix;
};


#endif

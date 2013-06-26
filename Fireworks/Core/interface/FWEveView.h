// -*- C++ -*-
//
// Package:     Core
// Class  :     FWEveView
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Thu Mar 16 14:11:32 CET 2010
// $Id: FWEveView.h,v 1.31 2013/04/24 04:08:45 amraktad Exp $
//


#ifndef Fireworks_Core_FWEveView_h
#define Fireworks_Core_FWEveView_h

// user include files
#include "Fireworks/Core/interface/FWViewBase.h"
#include "Fireworks/Core/interface/FWDoubleParameter.h"
#include "Fireworks/Core/interface/FWBoolParameter.h"
#include "Fireworks/Core/interface/FWLongParameter.h"
#include "Fireworks/Core/interface/FWEnumParameter.h"
#include "Fireworks/Core/interface/FWEvePtr.h"

// forward declarations
class TGLViewer;
class TGLOrthoCamera;
class TGLPerspectiveCamera;
class TGLCameraGuide;
class TGLAnnotation;
class TEveViewer;
class TEveElementList;
class TEveScene;
class TEveWindowSlot;
class TEveCaloViz;

class FWEventAnnotation;
class CmsAnnotation;
class FWViewContextMenuHandlerGL;
class FWColorManager;
class FWViewContext;
class ViewerParameterGUI;
class FWViewEnergyScale;
class ScaleAnnotation;
class FWViewEnergyScaleEditor;

namespace fireworks
{
   class Context;
}

class FWEveView : public FWViewBase
{
public:
   FWEveView(TEveWindowSlot*, FWViewType::EType, unsigned int version = 7);
   virtual ~FWEveView();

   virtual void setFrom(const FWConfiguration&);
   virtual void setBackgroundColor(Color_t);
   virtual void eventEnd();
   virtual void eventBegin();

   virtual void setContext(const fireworks::Context& x);
   const fireworks::Context& context()  { return *m_context; } 

   // ---------- const member functions --------------------- 

   virtual void addTo(FWConfiguration&) const;
   virtual FWViewContextMenuHandlerBase* contextMenuHandler() const;
   virtual void saveImageTo(const std::string& iName) const;
   virtual void populateController(ViewerParameterGUI&) const;

   TGLViewer*  viewerGL() const;
   TEveViewer* viewer()      { return m_viewer; }
   TEveScene*  eventScene()  { return m_eventScene;}
   TEveScene*  geoScene()    { return m_geoScene; }

   TEveElement*   ownedProducts()  { return m_ownedProducts; }
   FWViewContext* viewContext() { return m_viewContext.get(); }

   // ---------- static member functions --------------------
   virtual void useGlobalEnergyScaleChanged();
   virtual bool isEnergyScaleGlobal() const;
   virtual void setupEnergyScale();
   virtual void voteCaloMaxVal();

   virtual bool requestGLHandlerPick() const { return 0;} 
   
protected:
   virtual void resetCamera();
   virtual void pointLineScalesChanged();
   virtual void cameraGuideChanged();

   // scales
   virtual TEveCaloViz* getEveCalo() const { return 0; }

   // config
   void addToOrthoCamera(TGLOrthoCamera*, FWConfiguration&) const;
   void setFromOrthoCamera(TGLOrthoCamera*, const FWConfiguration&);
   void addToPerspectiveCamera(TGLPerspectiveCamera*, const std::string&, FWConfiguration&) const;
   void setFromPerspectiveCamera(TGLPerspectiveCamera*,  const std::string&, const FWConfiguration&);

private:
   FWEveView(const FWEveView&);    // stop default
   const FWEveView& operator=(const FWEveView&);    // stop default
  

   // ---------- member data --------------------------------

   TEveViewer*          m_viewer;
   TEveScene*           m_eventScene;
   TEveElement*         m_ownedProducts;
   TEveScene*           m_geoScene;

   FWEventAnnotation*   m_overlayEventInfo;  
   CmsAnnotation*       m_overlayLogo;
   ScaleAnnotation*     m_energyMaxValAnnotation;
   TGLCameraGuide*      m_cameraGuide;

   const fireworks::Context*  m_context;



private:
   // style parameters
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,26,0)
   FWDoubleParameter   m_imageScale;
#endif
   FWEnumParameter   m_eventInfoLevel;
   FWBoolParameter   m_drawCMSLogo;

   FWBoolParameter   m_pointSmooth;
   FWDoubleParameter m_pointSize;
   FWBoolParameter   m_lineSmooth; 
   FWDoubleParameter m_lineWidth;
   FWDoubleParameter m_lineOutlineScale;
   FWDoubleParameter m_lineWireframeScale;

   FWBoolParameter   m_showCameraGuide;
   FWBoolParameter   m_useGlobalEnergyScale;

   boost::shared_ptr<FWViewContextMenuHandlerGL>   m_viewContextMenu;
   std::auto_ptr<FWViewContext> m_viewContext;
   std::auto_ptr<FWViewEnergyScale> m_localEnergyScale;

   mutable FWViewEnergyScaleEditor* m_viewEnergyScaleEditor;
};


#endif

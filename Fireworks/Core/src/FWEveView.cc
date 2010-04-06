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
// $Id$
//



#include <RVersion.h>
#include <boost/bind.hpp>
#include <stdexcept>


// user include files

#define private public  //!!! TODO add get/sets for camera zoom and FOV
#include "TGLOrthoCamera.h"
#include "TGLPerspectiveCamera.h"
#undef private

#include "TGLEmbeddedViewer.h"
#include "TEveViewer.h"
#include "TGLScenePad.h"
#include "TEveManager.h"
#include "TEveElement.h"
#include "TEveWindow.h"
#include "TEveScene.h"

#include "Fireworks/Core/interface/FWEveView.h"
#include "Fireworks/Core/interface/FWEventAnnotation.h"
#include "Fireworks/Core/interface/CmsAnnotation.h"
#include "Fireworks/Core/interface/FWGLEventHandler.h"
#include "Fireworks/Core/interface/FWViewContextMenuHandlerGL.h"
#include "Fireworks/Core/interface/FWConfiguration.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/fwLog.h"

//
// constructors and destructor
//

FWEveView::FWEveView(TEveWindowSlot* iParent) :
   m_type(FWViewType::k3D),
   m_viewer(0),
   m_eventScene(0),
   m_geoScene(0),
   m_overlayEventInfo(0),
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,26,0)
   m_imageScale(this, "Image Scale", 1.0, 1.0, 6.0),
   m_eventInfoLevel(this, "Overlay Event Info", 0l, 0l, 3l),
   m_drawCMSLogo(this,"Show Logo",false)
#else
   m_eventInfoLevel(this, "Overlay Event Info", 0l, 0l, 3l),
   m_drawCMSLogo(this,"Show Logo",false),
   m_lineWidth(this,"Line width",1.0,1.0,10.0)
#endif
{
   m_viewer = new TEveViewer(typeName().c_str());

   TGLEmbeddedViewer* embeddedViewer;
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,25,4)
   embeddedViewer =  m_viewer->SpawnGLEmbeddedViewer(0);
#else
   embeddedViewer =  m_viewer->SpawnGLEmbeddedViewer();
#endif
   iParent->ReplaceWindow(m_viewer);
   gEve->GetViewers()->AddElement(m_viewer);
   // spawn geo scene
   m_geoScene = gEve->SpawnNewScene("Viewer GeoScene");
   m_geoScene->GetGLScene()->SetSelectable(kFALSE);
   m_viewer->AddScene(m_geoScene);
   m_viewContextMenu.reset(new FWViewContextMenuHandlerGL(m_viewer));

   FWGLEventHandler* eh = new FWGLEventHandler((TGWindow*)embeddedViewer->GetGLWidget(), (TObject*)embeddedViewer);
   embeddedViewer->SetEventHandler(eh);
   eh->openSelectedModelContextMenu_.connect(openSelectedModelContextMenu_);
   FWViewContextMenuHandlerGL* ctxHand = new FWViewContextMenuHandlerGL(m_viewer);
   ctxHand->setPickCameraCenter(true);
   m_viewContextMenu.reset(ctxHand);
   
   m_overlayEventInfo = new FWEventAnnotation(embeddedViewer);
   m_overlayEventInfo->setLevel(0);
   m_eventInfoLevel.changed_.connect(boost::bind(&FWEventAnnotation::setLevel,m_overlayEventInfo, _1));
   
   m_overlayLogo = new CmsAnnotation(embeddedViewer, 0.02, 0.98);
   m_overlayLogo->setVisible(false);
   m_drawCMSLogo.changed_.connect(boost::bind(&CmsAnnotation::setVisible,m_overlayLogo, _1));
 
#if ROOT_VERSION_CODE < ROOT_VERSION(5,26,0)  
   m_lineWidth.changed_.connect(boost::bind(&FWEveView::lineWidthChanged,this));
#endif
}

FWEveView::~FWEveView()
{
   if (m_geoScene) delete m_geoScene;
   m_viewer->DestroyWindowAndSlot();
}

//______________________________________________________________________________
// const member functions

const std::string& 
FWEveView::typeName() const
{
   return m_type.name();
}

FWViewContextMenuHandlerBase* 
FWEveView::contextMenuHandler() const {
   return (FWViewContextMenuHandlerBase*)m_viewContextMenu.get();
}

TGLViewer* 
FWEveView::viewerGL() const
{
   return  m_viewer->GetGLViewer();
}

void
FWEveView::saveImageTo(const std::string& iName) const
{
   bool succeeded = false;
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,26,0)
   succeeded = viewerGL()->SavePictureScale(iName, m_imageScale.value());
#else
   succeeded = viewerGL()->SavePicture(iName.c_str());
#endif

   if(!succeeded) {
      throw std::runtime_error("Unable to save picture");
   }
   fwLog(fwlog::kInfo) <<  "Saved image " << iName << std::endl;
}

//-------------------------------------------------------------------------------
void
FWEveView::lineWidthChanged()
{
#if ROOT_VERSION_CODE < ROOT_VERSION(5,26,0)
   viewerGL()->SetLineScale(m_lineWidth.value());
   viewerGL()->RequestDraw();
#endif
}

void
FWEveView::eventEnd()
{
   m_overlayEventInfo->setEvent();
}

void
FWEveView::setBackgroundColor(Color_t iColor)
{
   FWColorManager::setColorSetViewer(viewerGL(), iColor);
}

void
FWEveView::resetCamera()
{
   viewerGL()->ResetCurrentCamera();
}

void
FWEveView::setEventScene(TEveScene* eventScene)
{
   m_eventScene = eventScene;
}


void
FWEveView::setType(FWViewType::EType t)
{
   m_type = FWViewType(t);

   // update viewer name for debug purposes
   m_viewer->SetElementName(Form("Viewer_%s", typeName().c_str()));
}

//-------------------------------------------------------------------------------
void
FWEveView::addTo(FWConfiguration& iTo) const
{
   // take care of parameters
   FWConfigurableParameterizable::addTo(iTo);
   
   { 
      assert ( m_overlayEventInfo );
      m_overlayEventInfo->addTo(iTo);
   }
   { 
      assert ( m_overlayLogo );
      m_overlayLogo->addTo(iTo);
   }
}

void
FWEveView::setFrom(const FWConfiguration& iFrom)
{
   // take care of parameters
   FWConfigurableParameterizable::setFrom(iFrom);
   {
      assert( m_overlayEventInfo);
      m_overlayEventInfo->setFrom(iFrom);
   }
   {
      assert( m_overlayLogo);
      m_overlayLogo->setFrom(iFrom);
   }
}


void
FWEveView::addToOrthoCamera(TGLOrthoCamera* camera, FWConfiguration& iTo) const
{
   // zoom
   std::ostringstream s;
   s<<(camera->fZoom);
   std::string name("cameraZoom");
   iTo.addKeyValue(name+typeName(),FWConfiguration(s.str()));
   
   // transformation matrix
   std::string matrixName("cameraMatrix");
   for ( unsigned int i = 0; i < 16; ++i ) {
      std::ostringstream osIndex;
      osIndex << i;
      std::ostringstream osValue;
      osValue << camera->GetCamTrans()[i];
      iTo.addKeyValue(matrixName+osIndex.str()+typeName(),FWConfiguration(osValue.str()));
   }   
}

void
FWEveView::setFromOrthoCamera(TGLOrthoCamera* camera,  const FWConfiguration& iFrom)
{
   // zoom
   std::string zoomName("cameraZoom"); zoomName += typeName();
   assert( 0!=iFrom.valueForKey(zoomName) );
   std::istringstream s(iFrom.valueForKey(zoomName)->value());
   s>>(camera->fZoom);
   
   // transformation matrix
   std::string matrixName("cameraMatrix");
   for ( unsigned int i = 0; i < 16; ++i ) {
      std::ostringstream os;
      os << i;
      const FWConfiguration* value = iFrom.valueForKey( matrixName + os.str() + typeName() );
      assert( value );
      std::istringstream s(value->value());
      s>> (camera->RefCamTrans()[i]);
   }

   camera->IncTimeStamp();
}


 
void
FWEveView::addToPerspectiveCamera(TGLPerspectiveCamera* cam, const std::string& name, FWConfiguration& iTo) const
{   
   // transformation matrix
   std::string matrixName("cameraMatrix");
   for ( unsigned int i = 0; i < 16; ++i ){
      std::ostringstream osIndex;
      osIndex << i;
      std::ostringstream osValue;
      osValue << (cam->GetCamTrans())[i];
      iTo.addKeyValue(matrixName+osIndex.str()+name,FWConfiguration(osValue.str()));
   }
   
   // transformation matrix base
   matrixName = "cameraMatrixBase";
   for ( unsigned int i = 0; i < 16; ++i ){
      std::ostringstream osIndex;
      osIndex << i;
      std::ostringstream osValue;
      osValue << (cam->GetCamBase())[i];
      iTo.addKeyValue(matrixName+osIndex.str()+name,FWConfiguration(osValue.str()));
   }
   {
      std::ostringstream osValue;
      osValue << cam->fFOV;
      iTo.addKeyValue(name+" FOV",FWConfiguration(osValue.str()));
   }
   
}

void
FWEveView::setFromPerspectiveCamera(TGLPerspectiveCamera* cam, const std::string& name, const FWConfiguration& iFrom)
{
   std::string matrixName("cameraMatrix");
   for ( unsigned int i = 0; i < 16; ++i ){
      std::ostringstream os;
      os << i;
      const FWConfiguration* value = iFrom.valueForKey( matrixName + os.str() + name );
      assert( value );
      std::istringstream s(value->value());
      s>>((cam->RefCamTrans())[i]);
   }
   
   // transformation matrix base
   matrixName = "cameraMatrixBase";
   for ( unsigned int i = 0; i < 16; ++i ){
      std::ostringstream os;
      os << i;
      const FWConfiguration* value = iFrom.valueForKey( matrixName + os.str() + name );
      assert( value );
      std::istringstream s(value->value());
      s>>((cam->RefCamBase())[i]);
   }
   
   {
      const FWConfiguration* value = iFrom.valueForKey( name + " FOV" );
      if (value) // assert not necessary in version 1
      {
         std::istringstream s(value->value());
         s>>cam->fFOV;
      }
   }
   
   cam->IncTimeStamp();
}


//// TODO : add getters, setters for FOV and ZOOM in camera

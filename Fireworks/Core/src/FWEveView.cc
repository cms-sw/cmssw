#include <RVersion.h>
#include <boost/bind.hpp>
#include <stdexcept>


#include "TGLEmbeddedViewer.h"
#include "TEveViewer.h"
#include "TEveManager.h"
#include "TEveElement.h"
#include "TEveWindow.h"
#include "TEveScene.h"

// user include files
#include "Fireworks/Core/interface/FWEveView.h"

#include "Fireworks/Core/interface/FWEventAnnotation.h"
#include "Fireworks/Core/interface/CmsAnnotation.h"

#include "Fireworks/Core/interface/FWGLEventHandler.h"
#include "Fireworks/Core/interface/FWViewContextMenuHandlerGL.h"

#include "Fireworks/Core/interface/FWConfiguration.h"
#include "Fireworks/Core/interface/FWColorManager.h"


//
// constructors and destructor
//

FWEveView::FWEveView(TEveWindowSlot* iParent) :
   m_viewer(),
   m_scene(),
   m_overlayEventInfo(0),
   m_overlayEventInfoLevel(this, "Overlay Event Info", 0l, 0l, 3l),
   m_drawCMSLogo(this,"Show Logo",false),
   m_lineWidth(this,"Line width",1.0,1.0,10.0)
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

   m_scene = gEve->SpawnNewScene(typeName().c_str());
   m_viewer->AddScene(m_scene);

   m_viewContextMenu.reset(new FWViewContextMenuHandlerGL(m_viewer));

   FWGLEventHandler* eh = new FWGLEventHandler((TGWindow*)embeddedViewer->GetGLWidget(), (TObject*)embeddedViewer);
   embeddedViewer->SetEventHandler(eh);
   eh->openSelectedModelContextMenu_.connect(openSelectedModelContextMenu_);
   FWViewContextMenuHandlerGL* ctxHand = new FWViewContextMenuHandlerGL(m_viewer);
   ctxHand->setPickCameraCenter(true);
   m_viewContextMenu.reset(ctxHand);
   
   m_overlayEventInfo = new FWEventAnnotation(embeddedViewer);
   m_overlayEventInfoLevel.changed_.connect(boost::bind(&FWEventAnnotation::setLevel,m_overlayEventInfo, _1));
   
   m_overlayLogo = new CmsAnnotation(embeddedViewer, 0.02, 0.98);
   m_drawCMSLogo.changed_.connect(boost::bind(&CmsAnnotation::setVisible,m_overlayLogo, _1));
   
   m_lineWidth.changed_.connect(boost::bind(&FWEveView::lineWidthChanged,this));
}

FWEveView::~FWEveView()
{
   m_scene->Destroy();
   m_viewer->DestroyWindowAndSlot();
}

//______________________________________________________________________________
// const member functions

const std::string& 
FWEveView::typeName() const
{
   static std::string s_name("FWEveView");
   return s_name;
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
   bool succeeded = viewerGL()->SavePicture(iName.c_str());
   if(!succeeded) {
      throw std::runtime_error("Unable to save picture");
   }
}

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
FWEveView::lineWidthChanged()
{
   viewerGL()->SetLineScale(m_lineWidth.value());
   viewerGL()->RequestDraw();
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

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
//



#include <RVersion.h>
#include <boost/bind.hpp>
#include <stdexcept>


// user include files

#define private public  //!!! TODO add get/sets for camera zoom and FOV
#include "TGLOrthoCamera.h"
#include "TGLPerspectiveCamera.h"
#undef private
#include "TGLCameraGuide.h"

#include "TGLEmbeddedViewer.h"
#include "TGLScenePad.h"
#include "TEveManager.h"
#include "TEveElement.h"
#include "TEveWindow.h"
#include "TEveScene.h"
#define protected public  //!!! TODO add get/sets for TEveCalo2D for CellIDs
#include "TEveCalo.h"
#undef protected
#include "TGLOverlay.h"

#include "Fireworks/Core/interface/FWTEveViewer.h"
#include "Fireworks/Core/interface/FWTGLViewer.h"

#include "Fireworks/Core/interface/FWEveView.h"
#include "Fireworks/Core/interface/FWViewType.h"
#include "Fireworks/Core/interface/CmsShowViewPopup.h"
#include "Fireworks/Core/interface/FWEventAnnotation.h"
#include "Fireworks/Core/interface/CmsAnnotation.h"
#include "Fireworks/Core/interface/FWGLEventHandler.h"
#include "Fireworks/Core/interface/FWViewContextMenuHandlerGL.h"
#include "Fireworks/Core/interface/FWConfiguration.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWViewContext.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"
#include "Fireworks/Core/interface/CmsShowCommon.h"
#include "Fireworks/Core/interface/FWViewEnergyScaleEditor.h"

namespace fireworks
{
class Context;
}

/* This class is temporary workaround for missing in TGLAnnotation functionality */
class ScaleAnnotation : public TGLAnnotation
{
public:
   ScaleAnnotation(TGLViewerBase* parent, const char* text, Float_t posx, Float_t posy):
      TGLAnnotation(parent, text, posx, posy) {}
   virtual ~ScaleAnnotation() {}

   void setText(const char* txt)
   {
      fText = txt;
   }
};

//
// constructors and destructor
//

FWEveView::FWEveView(TEveWindowSlot* iParent, FWViewType::EType type, unsigned int version) :
   FWViewBase(type, version),
   m_context(0),
   m_viewer(0),
   m_eventScene(0),
   m_ownedProducts(0),
   m_geoScene(0),
   m_overlayEventInfo(0),
   m_overlayLogo(0),
   m_energyMaxValAnnotation(0),
   m_cameraGuide(0),
   // style
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,26,0)
   m_imageScale(this, "Image Scale", 1.0, 1.0, 6.0),
#endif
   m_eventInfoLevel(this, "Overlay Event Info", 0l, 0l, 2l),
   m_drawCMSLogo(this,"Show Logo",false),
   m_pointSmooth(this, "Smooth points", false),
   m_pointSize(this, "Point size", 1.0, 1.0, 10.0),
   m_lineSmooth(this, "Smooth lines", false),
   m_lineWidth(this,"Line width",1.0,1.0,10.0),
   m_lineOutlineScale(this, "Outline width scale", 1.0, 0.01, 10.0),
   m_lineWireframeScale(this, "Wireframe width scale", 1.0, 0.01, 10.0),
   m_showCameraGuide(this,"Show Camera Guide",false),
   m_useGlobalEnergyScale(this, "UseGlobalEnergyScale", true),
   m_viewContext( new FWViewContext()),
   m_localEnergyScale( new FWViewEnergyScale(FWViewType::idToName(type), version)),
   m_viewEnergyScaleEditor(0)
{
   m_viewer = new FWTEveViewer(typeName().c_str());

   FWTGLViewer *embeddedViewer = m_viewer->SpawnFWTGLViewer();
   iParent->ReplaceWindow(m_viewer);
   gEve->GetViewers()->AddElement(m_viewer);

   m_eventScene =  gEve->SpawnNewScene(Form("EventScene %s", typeName().c_str()));
   m_ownedProducts = new TEveElementList("ViewSpecificProducts");
   m_eventScene->AddElement(m_ownedProducts);

   m_viewer->AddScene(m_eventScene);

   // spawn geo scene
   m_geoScene = gEve->SpawnNewScene(Form("GeoScene %s", typeName().c_str()));
   m_geoScene->GetGLScene()->SetSelectable(kFALSE);
   m_viewer->AddScene(m_geoScene);

   FWGLEventHandler* eh = new FWGLEventHandler((TGWindow*)embeddedViewer->GetGLWidget(), (TObject*)embeddedViewer);
   embeddedViewer->SetEventHandler(eh);
   eh->setViewer(this);
   eh->openSelectedModelContextMenu_.connect(openSelectedModelContextMenu_);
   eh->SetDoInternalSelection(kFALSE);
   FWViewContextMenuHandlerGL* ctxHand = new FWViewContextMenuHandlerGL(this);
   // ctxHand->setPickCameraCenter(true);
   m_viewContextMenu.reset(ctxHand);
   
   m_energyMaxValAnnotation = new ScaleAnnotation(viewerGL(), "empty", 0.1, 0.9);
   m_energyMaxValAnnotation->SetRole(TGLOverlayElement::kViewer);
   m_energyMaxValAnnotation->SetState(TGLOverlayElement::kInvisible);
   m_energyMaxValAnnotation->SetUseColorSet(false);
   m_energyMaxValAnnotation->SetTextSize(0.05);
   m_energyMaxValAnnotation->SetTextColor(kMagenta);

   // style params

   m_overlayEventInfo = new FWEventAnnotation(embeddedViewer);
   m_overlayEventInfo->setLevel(0);

   m_eventInfoLevel.addEntry(0, "Nothing");
   m_eventInfoLevel.addEntry(1, "Run / event");
   m_eventInfoLevel.addEntry(2, "Run / event / lumi");
   m_eventInfoLevel.addEntry(3, "Full");
   m_eventInfoLevel.changed_.connect(boost::bind(&FWEventAnnotation::setLevel,m_overlayEventInfo, _1));
   
   m_overlayLogo = new CmsAnnotation(embeddedViewer, 0.02, 0.98);
   m_overlayLogo->setVisible(false);
   m_drawCMSLogo.changed_.connect(boost::bind(&CmsAnnotation::setVisible,m_overlayLogo, _1));

   m_cameraGuide = new TGLCameraGuide(0.9, 0.1, 0.08);
   m_cameraGuide->SetState(TGLOverlayElement::kInvisible);
   embeddedViewer->AddOverlayElement(m_cameraGuide);
   m_showCameraGuide.changed_.connect(boost::bind(&FWEveView::cameraGuideChanged,this));

   m_pointSmooth.changed_.connect(boost::bind(&FWEveView::pointLineScalesChanged,this));
   m_pointSize.changed_.connect(boost::bind(&FWEveView::pointLineScalesChanged,this));
   m_lineSmooth.changed_.connect(boost::bind(&FWEveView::pointLineScalesChanged,this));
   m_lineWidth.changed_.connect(boost::bind(&FWEveView::pointLineScalesChanged,this));
   m_lineOutlineScale.changed_.connect(boost::bind(&FWEveView::pointLineScalesChanged,this));
   m_lineWireframeScale.changed_.connect(boost::bind(&FWEveView::pointLineScalesChanged,this));


   // create scale for view  .. 
   m_viewContext->setEnergyScale(m_localEnergyScale.get());
   m_useGlobalEnergyScale.changed_.connect(boost::bind(&FWEveView::useGlobalEnergyScaleChanged, this));
   m_localEnergyScale->parameterChanged_.connect(boost::bind(&FWEveView::setupEnergyScale, this));
}

FWEveView::~FWEveView()
{
   m_geoScene->RemoveElements();
   m_eventScene->RemoveElements();
   m_viewer->DestroyWindowAndSlot();
}

//______________________________________________________________________________
// const member functions


FWViewContextMenuHandlerBase* 
FWEveView::contextMenuHandler() const {
   return dynamic_cast<FWViewContextMenuHandlerBase*> (m_viewContextMenu.get());
}

TGLViewer* 
FWEveView::viewerGL() const
{
   return  m_viewer->GetGLViewer();
}

TEveViewer*
FWEveView::viewer()
{
   return m_viewer;
}

FWTGLViewer* 
FWEveView::fwViewerGL() const
{
   return  m_viewer->fwGlViewer();
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
FWEveView::pointLineScalesChanged()
{
   viewerGL()->SetSmoothPoints(m_pointSmooth.value());
   viewerGL()->SetPointScale  (m_pointSize.value());
   viewerGL()->SetSmoothLines (m_lineSmooth.value());
   viewerGL()->SetLineScale   (m_lineWidth.value());
   viewerGL()->SetOLLineW     (m_lineOutlineScale.value());
   viewerGL()->SetWFLineW     (m_lineWireframeScale.value());
   viewerGL()->Changed();
   gEve->Redraw3D();
}

void
FWEveView::cameraGuideChanged()
{
   m_cameraGuide->SetBinaryState(m_showCameraGuide.value());
   viewerGL()->Changed();
   gEve->Redraw3D();
}

void
FWEveView::eventBegin()
{
}

void
FWEveView::eventEnd()
{
   m_overlayEventInfo->setEvent();
   setupEnergyScale();
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

//______________________________________________________________________________
void 
FWEveView::setContext(const fireworks::Context& x)
{
   m_context = &x ;

   // in constructor view context has local scale
   if (m_useGlobalEnergyScale.value()) 
      m_viewContext->setEnergyScale(context().commonPrefs()->getEnergyScale());
}

bool
FWEveView::isEnergyScaleGlobal() const
{
   return m_useGlobalEnergyScale.value();
}

void
FWEveView::useGlobalEnergyScaleChanged()
{
   m_viewContext->setEnergyScale(m_useGlobalEnergyScale.value() ? context().commonPrefs()->getEnergyScale() : m_localEnergyScale.get());
   if (m_viewEnergyScaleEditor) m_viewEnergyScaleEditor->setEnabled(!m_useGlobalEnergyScale.value());
   setupEnergyScale();
}

void
FWEveView::voteCaloMaxVal()
{
   TEveCaloViz* calo = getEveCalo();
   if (calo)
      context().voteMaxEtAndEnergy(calo->GetData()->GetMaxVal(1), calo->GetData()->GetMaxVal(0));
}

void
FWEveView::setupEnergyScale()
{
   // Called at end of event OR if scale parameters changed.

   FWViewEnergyScale*  energyScale = viewContext()->getEnergyScale();
   // printf("setupEnergyScale %s >> scale name %s\n", typeName().c_str(), energyScale->name().c_str());
   voteCaloMaxVal();
   
   // set cache for energy to lenght conversion
   float maxVal = context().getMaxEnergyInEvent(energyScale->getPlotEt());
   energyScale->updateScaleFactors(maxVal);
   // printf("max event val %f \n", maxVal);
   // printf("scales lego %f \n",  energyScale->getScaleFactorLego());

   // configure TEveCaloViz
   TEveCaloViz* calo = getEveCalo();
   if (calo)
   {
      calo->SetPlotEt(energyScale->getPlotEt());
      if (FWViewType::isLego(typeId()))
      {
         float f = energyScale->getScaleFactorLego();
         calo->SetMaxValAbs(TMath::Pi()/f);
      }
      else
      {
         float f = energyScale->getScaleFactor3D();
         calo->SetMaxValAbs(100/f);
      }
      calo->ElementChanged();
   }

   // emit signal to proxy builders 
   viewContext()->scaleChanged();
   gEve->Redraw3D();
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

   m_viewContext->getEnergyScale()->addTo(iTo);
}

void
FWEveView::setFrom(const FWConfiguration& iFrom)
{
   // Make sure you change the version ranges here
   // whenever you update the logic.
   // The rationale should be:
   // (version range supported by the next block) && (version range in the configuration file)
   //
   // This is not "forward" compatible, but I don't think
   // we care.
   if (version() >= 2 && iFrom.version() >= 1)
   {
      for(const_iterator it =begin(), itEnd = end();
          it != itEnd;
          ++it) {
         (*it)->setFrom(iFrom);      
      }  
   }
   if (iFrom.version() > 1)
   {
      assert( m_overlayEventInfo);
      m_overlayEventInfo->setFrom(iFrom);
   }
   {
      assert( m_overlayLogo);
      m_overlayLogo->setFrom(iFrom);
   }
   
   if (iFrom.version() > 4)
   {
      m_localEnergyScale->setFrom(iFrom);
   }


   // selection clors
   {
      const TGLColorSet& lcs = context().commonPrefs()->getLightColorSet();
      const TGLColorSet& dcs = context().commonPrefs()->getDarkColorSet();
      const UChar_t* ca = 0;

      ca = lcs.Selection(1).CArr();
      viewerGL()->RefLightColorSet().Selection(1).SetColor(ca[0], ca[1], ca[2]);
      ca = lcs.Selection(3).CArr();
      viewerGL()->RefLightColorSet().Selection(3).SetColor(ca[0], ca[1], ca[2]);
      ca = dcs.Selection(1).CArr();
      viewerGL()->RefDarkColorSet().Selection(1).SetColor(ca[0], ca[1], ca[2]);
      ca = dcs.Selection(3).CArr();
      viewerGL()->RefDarkColorSet().Selection(3).SetColor(ca[0], ca[1], ca[2]);
   }
}

//______________________________________________________________________________


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
   try {
      // zoom
      std::string zoomName("cameraZoom"); zoomName += typeName();
      if (iFrom.valueForKey(zoomName) == 0 )
      {
         throw std::runtime_error("can't restore parameter cameraZoom");
      }
      std::istringstream s(iFrom.valueForKey(zoomName)->value());
      s>>(camera->fZoom);
      
      // transformation matrix
      std::string matrixName("cameraMatrix");
      for ( unsigned int i = 0; i < 16; ++i ) {
         std::ostringstream os;
         os << i;
         const FWConfiguration* value = iFrom.valueForKey( matrixName + os.str() + typeName() );
         if ( value ==  0 )
         {
            throw std::runtime_error ("can't restore parameter cameraMatrix.");
         }
         std::istringstream s(value->value());
         s>> (camera->RefCamTrans()[i]);
      }
   }
   catch (const std::runtime_error& iException)
   {
      fwLog(fwlog::kInfo) << "Caught exception while restoring camera parameters in view " << typeName() << "\n.";
      viewerGL()->ResetCamerasAfterNextUpdate();      

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
   try {
      std::string matrixName("cameraMatrix");
      for ( unsigned int i = 0; i < 16; ++i ){
         std::ostringstream os;
         os << i;
         const FWConfiguration* value = iFrom.valueForKey( matrixName + os.str() + name );
         if ( value ==  0 )
         {
            throw std::runtime_error ("can't restore parameter cameraMatrix.");
         }
         std::istringstream s(value->value());
         s>>((cam->RefCamTrans())[i]);
      }
      
      // transformation matrix base
      matrixName = "cameraMatrixBase";
      for ( unsigned int i = 0; i < 16; ++i ){
         std::ostringstream os;
         os << i;
         const FWConfiguration* value = iFrom.valueForKey( matrixName + os.str() + name );
         if ( value ==  0 )
         {
            throw std::runtime_error ("can't restore parameter cameraMatrixBase.");
         }
     
         std::istringstream s(value->value());
         s>>((cam->RefCamBase())[i]);
      }
      
      {
         const FWConfiguration* value = iFrom.valueForKey( name + " FOV" );
         if ( value ==  0 )
         {
            throw std::runtime_error ("can't restore parameter cameraMatrixBase.");
         }
         std::istringstream s(value->value());
         s>>cam->fFOV;
      }
      
      cam->IncTimeStamp();
   }
   catch (const std::runtime_error& iException)
   {
      fwLog(fwlog::kInfo) << "Caught exception while restoring camera parameters in view " << typeName() << "\n.";
      viewerGL()->ResetCamerasAfterNextUpdate();  
      fwLog(fwlog::kDebug) << "Reset camera fo view "  << typeName() << "\n.";
   }
}


void 
FWEveView::populateController(ViewerParameterGUI& gui) const
{
   gui.requestTab("Style").
      addParam(&m_eventInfoLevel).
      addParam(&m_drawCMSLogo).
      addParam(&m_showCameraGuide).
      separator().
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,26,0)
      addParam(&m_imageScale).
#endif
      addParam(&m_pointSize).
      addParam(&m_pointSmooth).
      addParam(&m_lineSmooth).
      addParam(&m_lineWidth).
      addParam(&m_lineOutlineScale).
      addParam(&m_lineWireframeScale);


   gui.requestTab("Scales").
      addParam(&m_useGlobalEnergyScale);

   m_viewEnergyScaleEditor = new FWViewEnergyScaleEditor(m_localEnergyScale.get(), gui.getTabContainer(), !FWViewType::isLego(typeId()));
   m_viewEnergyScaleEditor->setEnabled(!m_useGlobalEnergyScale.value());
   gui.addFrameToContainer(m_viewEnergyScaleEditor);
}

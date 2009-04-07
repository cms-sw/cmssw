
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWEveLegoView
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 21 11:22:41 EST 2008
// $Id: FWEveLegoView.cc,v 1.38 2009/03/20 17:34:56 amraktad Exp $
//

// system include files
#include <algorithm>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/numeric/conversion/converter.hpp>
#include <iostream>
#include <sstream>
#include <stdexcept>

#define private public
#include "TGLOrthoCamera.h"
#undef private

#define protected public
#include "TEveLegoEventHandler.h"
#undef protected

#include "TRootEmbeddedCanvas.h"
#include "THStack.h"
#include "TCanvas.h"
#include "TClass.h"
#include "TH2F.h"
#include "TView.h"
#include "TColor.h"
#include "TEveScene.h"
#include "TGLViewer.h"
//EVIL, but only way I can avoid a double delete of TGLEmbeddedViewer::fFrame
#define private public
#include "TGLEmbeddedViewer.h"
#undef private
#include "TEveViewer.h"
#include "TEveManager.h"
#include "TEveElement.h"
#include "TEveCalo.h"
#include "TEveCaloData.h"
#include "TEveElement.h"
#include "TEveRGBAPalette.h"
#include "TGLPerspectiveCamera.h"
#include "TGLWidget.h"
#include "TEveTrans.h"
#include "TEveStraightLineSet.h"
#include "TEveCaloLegoOverlay.h"
#include "TEveWindow.h"

// user include files
#include "Fireworks/Core/interface/FWEveLegoView.h"
#include "Fireworks/Core/interface/FWConfiguration.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWEveLegoView::FWEveLegoView(TEveWindowSlot* iParent, TEveElementList* list) :
   //m_minEcalEnergy(this,"ECAL energy threshold (GeV)",1.,0.,100.),
   //m_minHcalEnergy(this,"HCAL energy threshold (GeV)",1.,0.,100.),
   //m_ecalSlice(0),
   //m_hcalSlice(0),
   m_lego(0),
   m_autoRebin(this,"Auto rebin on zoom",true),
   m_cameraMatrix(0),
   m_cameraMatrixBase(0),
   m_cameraMatrixRef(0),
   m_cameraMatrixBaseRef(0),
   m_orthoCameraZoom(0),
   m_orthoCameraMatrix(0),
   m_orthoCameraZoomRef(0),
   m_orthoCameraMatrixRef(0),
   m_topView(false),
   m_cameraSet(false)
{
   TEveViewer* nv = new TEveViewer(staticTypeName().c_str());
   m_embeddedViewer =  nv->SpawnGLEmbeddedViewer();
   iParent->ReplaceWindow(nv);

   TGLEmbeddedViewer* ev = m_embeddedViewer;

   m_autoRebin.changed_.connect(boost::bind(&FWEveLegoView::setAutoRebin,this));

   TEveScene* ns = gEve->SpawnNewScene(staticTypeName().c_str());
   m_scene.reset(ns);
   nv->AddScene(ns);
   m_viewer.reset(nv);
   gEve->AddElement(nv, gEve->GetViewers());

   // take care of cameras
   //
   ev->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
   //  ev->SetCurrentCamera(TGLViewer::kCameraPerspXOY);
   TEveLegoEventHandler* eh = new TEveLegoEventHandler("Lego", ev->GetGLWidget(), ev);
   eh->fMode = TEveLegoEventHandler::kLocked;
   ev->SetEventHandler(eh);


   if ( TGLPerspectiveCamera* camera = dynamic_cast<TGLPerspectiveCamera*>( &(ev->RefCamera(TGLViewer::kCameraPerspXOY) ))) {
      m_cameraMatrixRef = const_cast<TGLMatrix*>(&(camera->GetCamTrans()));
      m_cameraMatrixBaseRef = const_cast<TGLMatrix*>(&(camera->GetCamBase()));
   }
   if ( TGLOrthoCamera* camera = dynamic_cast<TGLOrthoCamera*>( &(ev->RefCamera(TGLViewer::kCameraOrthoXOY) ))) {
      m_orthoCameraZoomRef = &(camera->fZoom);
      m_orthoCameraMatrixRef = const_cast<TGLMatrix*>(&(camera->GetCamTrans()));
   }

  
   /*
     TEveRGBAPalette* pal = new TEveRGBAPalette(0, 100);
     // pal->SetLimits(0, data->GetMaxVal());
     pal->SetLimits(0, 100);
     pal->SetDefaultColor((Color_t)1000);

     m_lego = new TEveCaloLego();
     m_lego->InitMainTrans();
     m_lego->RefMainTrans().SetScale(2*M_PI, 2*M_PI, M_PI);
     m_lego->SetTopViewTowerColor(kWhite);

     m_lego->SetPalette(pal);
     // m_lego->SetMainColor(Color_t(TColor::GetColor("#0A0A0A")));
     m_lego->SetGridColor(Color_t(TColor::GetColor("#202020")));
     m_lego->Set2DMode(TEveCaloLego::kValSize);
     m_lego->SetBinWidth(6);
     // add calorimeter boundaries
     TEveStraightLineSet* boundaries = new TEveStraightLineSet("boundaries");
     boundaries->SetPickable(kFALSE);
     boundaries->SetLineColor(Color_t(TColor::GetColor("#404040")));
     // boundaries->SetLineWidth(2);
     boundaries->AddLine(-1.479,-3.1416,0.001,-1.479,3.1416,0.001);
     boundaries->AddLine(1.479,-3.1416,0.001,1.479,3.1416,0.001);
     boundaries->AddLine(-3.0,-3.1416,0.001,-3.0,3.1416,0.001);
     boundaries->AddLine(3.0,-3.1416,0.001,3.0,3.1416,0.001);
     gEve->AddElement(boundaries,m_lego);
     // lego->SetEtaLimits(etaLimLow, etaLimHigh);
     // lego->SetTitle("caloTower Et distribution");
     gEve->AddElement(m_lego, ns);
     gEve->AddToListTree(m_lego, kTRUE);
   */
   gEve->AddElement(list,ns);
   gEve->AddToListTree(list, kTRUE);
   //CDJ This is done too early setCameras();
   //m_minEcalEnergy.changed_.connect(boost::bind(&FWEveLegoView::setMinEcalEnergy,this,_1));
   //m_minHcalEnergy.changed_.connect(boost::bind(&FWEveLegoView::setMinHcalEnergy,this,_1));
   if (list->HasChildren())
   {
      m_lego =  dynamic_cast<TEveCaloLego*>( list->FirstChild());
      if (m_lego) {
         TEveCaloLegoOverlay* overlay = new TEveCaloLegoOverlay();
         overlay->SetShowPlane(kFALSE);
         overlay->SetShowPerspective(kFALSE);
         overlay->GetAttAxis()->SetLabelSize(0.02);
         overlay->GetAttAxis()->SetLabelColor(kWhite);
         ev->AddOverlayElement(overlay);
         overlay->SetCaloLego(m_lego);
         gEve->AddElement(overlay, ns);
      }
   }
   setCameras();
}

FWEveLegoView::~FWEveLegoView()
{
   m_scene->Destroy();
   m_viewer->DestroyWindowAndSlot();

   m_viewer.destroyElement();
   delete m_cameraMatrix;
   delete m_cameraMatrixBase;
   delete m_orthoCameraMatrix;
}

void
FWEveLegoView::finishSetup()
{
   if ( !m_cameraSet ) setCameras();
}

void
FWEveLegoView::setBackgroundColor(Color_t iColor)
{
   m_viewer->GetGLViewer()->SetClearColor(iColor);
}

void
FWEveLegoView::setCameras()
{
   // Few words on what is going on. First we paint the scene (not
   // sure it's needed).  Than we redraw everything with a lego
   // object already projected, reseting all the cameras. If
   // parameters were set from a config file, apply them directly to
   // the cameras. Add a small negative rotation (a kludgey
   // solution), to cause decrease in theta angle of the view to
   // emulate conditions similar to what happens during transition
   // from 3D to top 2D view.
   m_scene->Repaint();
   m_viewer->Redraw(kTRUE);
   if ( m_cameraMatrix && m_cameraMatrixBase && m_orthoCameraMatrix) {
      *m_cameraMatrixRef = *m_cameraMatrix;
      *m_cameraMatrixBaseRef = *m_cameraMatrixBase;
      *m_orthoCameraMatrixRef = *m_orthoCameraMatrix;
      *m_orthoCameraZoomRef = m_orthoCameraZoom;
      TEveLegoEventHandler* eh =
         dynamic_cast<TEveLegoEventHandler*>(m_viewer->GetGLViewer()->GetEventHandler());
      if ( m_topView && eh ) {
         eh->Rotate(0,10000,kFALSE, kFALSE);
      }
   }
   m_cameraSet = true;
}

#if defined(THIS_WILL_NEVER_BE_DEFINED)
void
FWEveLegoView::draw(TEveCaloDataHist* data)
{
   // bool firstTime = (m_lego->GetData() == 0);
   m_lego->SetData(data);
   m_lego->ElementChanged();
   m_lego->DataChanged();
   if ( !m_cameraSet ) setCameras();*/
   /*
      {
      m_scene->Repaint();
      m_viewer->Redraw(kTRUE);
      // std::cout << "Viewer: " <<  m_viewer << std::endl;
      // m_viewer->GetGLViewer()->ResetCameras();
      m_cameraSet = true;
      }
    */
   // m_viewer->GetGLViewer()->UpdateScene();
   //CDJ m_viewer->GetGLViewer()->RequestDraw();
}

void
FWEveLegoView::setMinEcalEnergy(double value)
{
   /*
      const std::string name = "ecalLego";
      if ( ! m_lego->GetData() ) return;
      if ( ! m_ecalSlice )
      for ( int i = 0; i < m_lego->GetData()->GetNSlices(); ++i )
       if ( name == m_lego->GetData()->RefSliceInfo(i).fHist->GetName() )
         {
            m_ecalSlice = &(m_lego->GetData()->RefSliceInfo(i));
            break;
         }
      if ( ! m_ecalSlice ) return;
      m_ecalSlice->fThreshold = value;
      m_lego->ElementChanged();
      m_lego->DataChanged();
      m_viewer->GetGLViewer()->RequestDraw();
    */
}

void
FWEveLegoView::setMinHcalEnergy(double value)
{
   /*
      const std::string name = "hcalLego";
      if ( ! m_lego->GetData() ) return;
      if ( ! m_hcalSlice )
      for ( int i = 0; i < m_lego->GetData()->GetNSlices(); ++i )
       if ( name == m_lego->GetData()->RefSliceInfo(i).fHist->GetName() )
         {
            m_hcalSlice = &(m_lego->GetData()->RefSliceInfo(i));
            break;
         }
      if ( ! m_hcalSlice ) return;
      m_hcalSlice->fThreshold = value;
      m_lego->ElementChanged();
      m_lego->DataChanged();
      m_viewer->GetGLViewer()->RequestDraw();
    */
}

void
FWEveLegoView::setMinEnergy()
{
   /*
      setMinEcalEnergy( m_minEcalEnergy.value() );
      setMinHcalEnergy( m_minHcalEnergy.value() );
    */
}
#endif

void
FWEveLegoView::setFrom(const FWConfiguration& iFrom)
{
   // take care of parameters
   FWConfigurableParameterizable::setFrom(iFrom);

   // retrieve camera parameters
   m_cameraMatrix = new TGLMatrix();
   m_cameraMatrixBase = new TGLMatrix();
   m_orthoCameraMatrix = new TGLMatrix();

/*   // state
   std::string stateName("cameraState"); stateName += typeName();
   assert( 0!=iFrom.valueForKey(stateName) );
   std::istringstream s(iFrom.valueForKey(stateName)->value());
   bool cameraState;
   s>>cameraState;
   if ( cameraState )
     m_embeddedViewer->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
   else
     m_embeddedViewer->SetCurrentCamera(TGLViewer::kCameraPerspXOY);
   // m_embeddedViewer->ResetCurrentCamera();
 */
// transformation matrix
   assert(m_cameraMatrix);
   std::string matrixName("cameraMatrix");
   for ( unsigned int i = 0; i < 16; ++i ){
      std::ostringstream os;
      os << i;
      const FWConfiguration* value = iFrom.valueForKey( matrixName + os.str() + "Lego" );
      assert( value );
      std::istringstream s(value->value());
      s>>((*m_cameraMatrix)[i]);
   }

   // transformation matrix base
   assert(m_cameraMatrixBase);
   matrixName = "cameraMatrixBase";
   for ( unsigned int i = 0; i < 16; ++i ){
      std::ostringstream os;
      os << i;
      const FWConfiguration* value = iFrom.valueForKey( matrixName + os.str() + "Lego" );
      assert( value );
      std::istringstream s(value->value());
      s>>((*m_cameraMatrixBase)[i]);
   }

   // zoom
   std::string zoomName("orthoCameraZoom"); zoomName += typeName();
   assert( 0!=iFrom.valueForKey(zoomName) );
   std::istringstream s(iFrom.valueForKey(zoomName)->value());
   s>>(m_orthoCameraZoom);

   // transformation matrix
   assert(m_orthoCameraMatrix);
   std::string orthoMatrixName("orthoCameraMatrix");
   for ( unsigned int i = 0; i < 16; ++i ){
      std::ostringstream os;
      os << i;
      const FWConfiguration* value = iFrom.valueForKey( orthoMatrixName + os.str() + typeName() );
      assert( value );
      std::istringstream s(value->value());
      s>>((*m_orthoCameraMatrix)[i]);
   }

   // topView
   {
      std::string stateName("topView"); stateName += typeName();
      assert( 0!=iFrom.valueForKey(stateName) );
      std::istringstream s(iFrom.valueForKey(stateName)->value());
      s >> m_topView;
   }
}

void 
FWEveLegoView::setAutoRebin()
{
   if(m_lego) {
      m_lego->SetAutoRebin(m_autoRebin.value());
      m_lego->ElementChanged(kTRUE,kTRUE);
   }
}


//
// const member functions
//
TGFrame*
FWEveLegoView::frame() const
{
   return m_embeddedViewer->GetFrame();
}

const std::string&
FWEveLegoView::typeName() const
{
   return staticTypeName();
}

void
FWEveLegoView::addTo(FWConfiguration& iTo) const
{
   // take care of parameters
   FWConfigurableParameterizable::addTo(iTo);

   // store camera parameters

   // transformation matrix
   assert(m_cameraMatrixRef);
   std::string matrixName("cameraMatrix");
   for ( unsigned int i = 0; i < 16; ++i ){
      std::ostringstream osIndex;
      osIndex << i;
      std::ostringstream osValue;
      osValue << (*m_cameraMatrixRef)[i];
      iTo.addKeyValue(matrixName+osIndex.str()+"Lego",FWConfiguration(osValue.str()));
   }

   // transformation matrix base
   assert(m_cameraMatrixBaseRef);
   matrixName = "cameraMatrixBase";
   for ( unsigned int i = 0; i < 16; ++i ){
      std::ostringstream osIndex;
      osIndex << i;
      std::ostringstream osValue;
      osValue << (*m_cameraMatrixBaseRef)[i];
      iTo.addKeyValue(matrixName+osIndex.str()+"Lego",FWConfiguration(osValue.str()));
   }

   // zoom
   assert(m_orthoCameraZoomRef);
   std::ostringstream s;
   s<<(*m_orthoCameraZoomRef);
   std::string name("orthoCameraZoom");
   iTo.addKeyValue(name+typeName(),FWConfiguration(s.str()));

   // zoom
   s.str("");
   bool topView = false;
   if ( dynamic_cast<TGLOrthoCamera*>( &(m_embeddedViewer->CurrentCamera()) ) )
      topView = true;
   s << topView;
   name = "topView";
   iTo.addKeyValue(name+typeName(),FWConfiguration(s.str()));

   // transformation matrix
   assert(m_orthoCameraMatrixRef);
   std::string orthoMatrixName("orthoCameraMatrix");
   for ( unsigned int i = 0; i < 16; ++i ){
      std::ostringstream osIndex;
      osIndex << i;
      std::ostringstream osValue;
      osValue << (*m_orthoCameraMatrixRef)[i];
      iTo.addKeyValue(orthoMatrixName+osIndex.str()+typeName(),FWConfiguration(osValue.str()));
   }
}

void
FWEveLegoView::saveImageTo(const std::string& iName) const
{
   bool succeeded = m_viewer->GetGLViewer()->SavePicture(iName.c_str());
   if(!succeeded) {
      throw std::runtime_error("Unable to save picture");
   }
}


//
// static member functions
//
const std::string&
FWEveLegoView::staticTypeName()
{
   static std::string s_name("3D Lego");
   return s_name;
}


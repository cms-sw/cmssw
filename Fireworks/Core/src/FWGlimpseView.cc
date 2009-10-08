// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGlimpseView
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 21 11:22:41 EST 2008
// $Id: FWGlimpseView.cc,v 1.32 2009/10/06 11:26:22 amraktad Exp $
//

// system include files
#include <algorithm>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/numeric/conversion/converter.hpp>
#include <iostream>
#include <sstream>
#include <stdexcept>

// FIXME
// need camera parameters
#define private public
#include "TGLPerspectiveCamera.h"
#undef private


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
#include "TEveWindow.h"
#include "TEveElement.h"
#include "TEveCalo.h"
#include "TEveElement.h"
#include "TEveLegoEventHandler.h"
#include "TGLWidget.h"
#include "TGLScenePad.h"
#include "TGLFontManager.h"
#include "TEveTrans.h"
#include "TGeoTube.h"
#include "TEveGeoNode.h"
#include "TEveStraightLineSet.h"
#include "TEveText.h"
#include "TGeoArb8.h"

// user include files
#include "Fireworks/Core/interface/FWGLEventHandler.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/FWGlimpseView.h"
#include "Fireworks/Core/interface/FWEveValueScaler.h"
#include "Fireworks/Core/interface/FWConfiguration.h"
#include "Fireworks/Core/interface/BuilderUtils.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//
//double FWGlimpseView::m_scale = 1;

//
// constructors and destructor
//
FWGlimpseView::FWGlimpseView(TEveWindowSlot* iParent, TEveElementList* list,
                             FWEveValueScaler* iScaler) :
   m_cylinder(0),
   m_cameraMatrix(0),
   m_cameraMatrixBase(0),
   m_cameraFOV(0),
   // m_scaleParam(this,"Energy scale", static_cast<double>(iScaler->scale()), 0.01, 1000.),
   m_showAxes(this, "Show Axes", true ),
   m_showCylinder(this, "Show Cylinder", true),
   m_scaler(iScaler)
{
   TEveViewer* nv = new TEveViewer(staticTypeName().c_str());
   m_embeddedViewer =  nv->SpawnGLEmbeddedViewer();
   iParent->ReplaceWindow(nv);

   FWGLEventHandler* eh = new FWGLEventHandler((TGWindow*)m_embeddedViewer->GetGLWidget(), (TObject*)m_embeddedViewer);
   m_embeddedViewer->SetEventHandler(eh);
   eh->openSelectedModelContextMenu_.connect(openSelectedModelContextMenu_);


   TGLEmbeddedViewer* ev = m_embeddedViewer;
   ev->SetCurrentCamera(TGLViewer::kCameraPerspXOZ);
   //? ev->SetEventHandler(new TGlimpseEventHandler("Lego", ev->GetGLWidget(), ev));
   m_cameraMatrix = const_cast<TGLMatrix*>(&(ev->CurrentCamera().GetCamTrans()));
   m_cameraMatrixBase = const_cast<TGLMatrix*>(&(ev->CurrentCamera().GetCamBase()));
   if ( TGLPerspectiveCamera* camera =
        dynamic_cast<TGLPerspectiveCamera*>(&(ev->CurrentCamera())) )
      m_cameraFOV = &(camera->fFOV);

   TEveScene* ns = gEve->SpawnNewScene(staticTypeName().c_str());
   m_scene = ns;
   nv->AddScene(ns);
   m_viewer=nv;
   gEve->AddElement(nv, gEve->GetViewers());
   gEve->AddElement(list,ns);
   gEve->AddToListTree(list, kTRUE);


   // create 3D axes
   TGLFont::EMode fontMode = TGLFont::kPixmap;
   Int_t fs = 14;
   Color_t fcol = kGray+1;

   // X axis
   TEveStraightLineSet* xAxis = new TEveStraightLineSet( "GlimpseXAxis" );
   xAxis->SetPickable(kTRUE);
   xAxis->SetTitle("Energy Scale, 100 GeV, X-axis (LHC center)");
   xAxis->SetLineStyle(3);
   xAxis->SetLineColor(fcol);
   xAxis->AddLine(-100,0,0,100,0,0);
   gEve->AddElement(xAxis, ns);

   TEveText* xTxt = new TEveText( "X+" );
   xTxt->PtrMainTrans()->SetPos(100-fs, -fs, 0);
   xTxt->SetFontMode(fontMode);
   xTxt->SetMainColor(fcol);
   gEve->AddElement(xTxt, ns);

   // Y axis
   TEveStraightLineSet* yAxis = new TEveStraightLineSet( "GlimpseYAxis" );
   yAxis->SetPickable(kTRUE);
   yAxis->SetTitle("Energy Scale, 100 GeV, Y-axis (upward)");
   yAxis->SetLineColor(fcol);
   yAxis->SetLineStyle(3);
   yAxis->AddLine(0,-100,0,0,100,0);
   gEve->AddElement(yAxis, ns);

   TEveText* yTxt = new TEveText( "Y+" );
   yTxt->PtrMainTrans()->SetPos(0, 100-fs, 0);
   yTxt->SetFontMode(fontMode);
   yTxt->SetMainColor(fcol);
   gEve->AddElement(yTxt, ns);

   // Z axis
   TEveStraightLineSet* zAxis = new TEveStraightLineSet( "GlimpseZAxis" );
   zAxis->SetPickable(kTRUE);
   zAxis->SetTitle("Energy Scale, 100 GeV, Z-axis (west, along beam)");
   zAxis->SetLineColor(fcol);
   zAxis->AddLine(0,0,-100,0,0,100);
   gEve->AddElement(zAxis, ns);

   TEveText* zTxt = new TEveText( "Z+" );
   zTxt->PtrMainTrans()->SetPos(0, -fs,  100 - zTxt->GetExtrude()*2);
   zTxt->SetFontMode(fontMode);
   zTxt->SetMainColor(fcol);
   gEve->AddElement(zTxt, ns);


   // made detector outline in wireframe scene

   TEveScene* wns = gEve->SpawnNewScene(Form("Wireframe %s", staticTypeName().c_str()));
   nv->AddScene(wns);
   TGLScene* gls  = wns->GetGLScene();
   gls->SetStyle(TGLRnrCtx::kWireFrame);
   gls->SetLOD(TGLRnrCtx::kLODMed);
   gls->SetSelectable(kFALSE);

   TEveGeoManagerHolder gmgr(TEveGeoShape::GetGeoMangeur());
   TGeoTube* tube = new TGeoTube(129,130,310);
   m_cylinder = fw::getShape("Detector outline", tube, kWhite);
   m_cylinder->SetPickable(kFALSE);
   m_cylinder->SetMainColor(kGray+3);
   wns->AddElement(m_cylinder);

   /*
     TGeoTrap* cube = new TGeoTrap(100,0,0,100,100,100,0,100,100,100,0);
     TEveGeoShapeExtract* extract = fw::getShapeExtract("Detector outline", cube, Color_t(TColor::GetColor("#202020")) );
     TEveElement* element = TEveGeoShape::ImportShapeExtract(extract, ns);
     element->SetPickable(kFALSE);
     element->SetMainTransparency(80);
     gEve->AddElement(element, ns);
   */

   /*
     TEveStraightLineSet* outline = new TEveStraightLineSet( "EnergyScale" );
     outline->SetPickable(kTRUE);
     outline->SetTitle("100 GeV Energy Scale Cube");
     double size = 100;
     outline->SetLineColor( Color_t(TColor::GetColor("#202020")) );
     outline->AddLine(-size, -size, -size,  size, -size, -size);
     outline->AddLine( size, -size, -size,  size,  size, -size);
     outline->AddLine( size,  size, -size, -size,  size, -size);
     outline->AddLine(-size,  size, -size, -size, -size, -size);
     outline->AddLine(-size, -size,  size,  size, -size,  size);
     outline->AddLine( size, -size,  size,  size,  size,  size);
     outline->AddLine( size,  size,  size, -size,  size,  size);
     outline->AddLine(-size,  size,  size, -size, -size,  size);
     outline->AddLine(-size, -size, -size, -size, -size,  size);
     outline->AddLine(-size,  size, -size, -size,  size,  size);
     outline->AddLine( size, -size, -size,  size, -size,  size);
     outline->AddLine( size,  size, -size,  size,  size,  size);
     gEve->AddElement(outline, ns);
   */
   // m_scaleParam.changed_.connect(boost::bind(&FWGlimpseView::updateScale,this,_1));
   m_showAxes.changed_.connect(boost::bind(&FWGlimpseView::showAxes,this));
   m_showCylinder.changed_.connect(boost::bind(&FWGlimpseView::showCylinder,this));
}

FWGlimpseView::~FWGlimpseView()
{
   m_scene->Destroy();
   m_viewer->DestroyWindowAndSlot();
}

void
FWGlimpseView::setFrom(const FWConfiguration& iFrom)
{
   // take care of parameters
   FWConfigurableParameterizable::setFrom(iFrom);

   // retrieve camera parameters

   // transformation matrix
   assert(m_cameraMatrix);
   std::string matrixName("cameraMatrix");
   for ( unsigned int i = 0; i < 16; ++i ){
      std::ostringstream os;
      os << i;
      const FWConfiguration* value = iFrom.valueForKey( matrixName + os.str() + "Glimpse" );
      if (!value ) continue;
      std::istringstream s(value->value());
      s>>((*m_cameraMatrix)[i]);
   }

   // transformation matrix base
   assert(m_cameraMatrixBase);
   matrixName = "cameraMatrixBase";
   for ( unsigned int i = 0; i < 16; ++i ){
      std::ostringstream os;
      os << i;
      const FWConfiguration* value = iFrom.valueForKey( matrixName + os.str() + "Glimpse" );
      if (!value ) continue;
      std::istringstream s(value->value());
      s>>((*m_cameraMatrixBase)[i]);
   }

   {
      assert ( m_cameraFOV );
      const FWConfiguration* value = iFrom.valueForKey( "Glimpse FOV" );
      if ( value ) {
         std::istringstream s(value->value());
         s>>*m_cameraFOV;
      }
   }
   m_viewer->GetGLViewer()->RequestDraw();
}

void
FWGlimpseView::setBackgroundColor(Color_t iColor) {
   FWColorManager::setColorSetViewer(m_viewer->GetGLViewer(), iColor);
}

//
// const member functions
//
TGFrame*
FWGlimpseView::frame() const
{
   return m_embeddedViewer->GetFrame();
}

const std::string&
FWGlimpseView::typeName() const
{
   return staticTypeName();
}

void
FWGlimpseView::addTo(FWConfiguration& iTo) const
{
   // take care of parameters
   FWConfigurableParameterizable::addTo(iTo);

   // store camera parameters

   // transformation matrix
   assert(m_cameraMatrix);
   std::string matrixName("cameraMatrix");
   for ( unsigned int i = 0; i < 16; ++i ){
      std::ostringstream osIndex;
      osIndex << i;
      std::ostringstream osValue;
      osValue << (*m_cameraMatrix)[i];
      iTo.addKeyValue(matrixName+osIndex.str()+"Glimpse",FWConfiguration(osValue.str()));
   }

   // transformation matrix base
   assert(m_cameraMatrixBase);
   matrixName = "cameraMatrixBase";
   for ( unsigned int i = 0; i < 16; ++i ){
      std::ostringstream osIndex;
      osIndex << i;
      std::ostringstream osValue;
      osValue << (*m_cameraMatrixBase)[i];
      iTo.addKeyValue(matrixName+osIndex.str()+"Glimpse",FWConfiguration(osValue.str()));
   }
   {
      assert ( m_cameraFOV );
      std::ostringstream osValue;
      osValue << *m_cameraFOV;
      iTo.addKeyValue("Glimpse FOV",FWConfiguration(osValue.str()));
   }
}

void
FWGlimpseView::saveImageTo(const std::string& iName) const
{
   bool succeeded = m_viewer->GetGLViewer()->SavePicture(iName.c_str());
   if(!succeeded) {
      throw std::runtime_error("Unable to save picture");
   }
}

void
FWGlimpseView::updateScale( double scale )
{
   m_scaler->setScale(scale);
}

void
FWGlimpseView::showAxes( )
{
   if ( m_showAxes.value() )
      m_embeddedViewer->SetGuideState(TGLUtil::kAxesOrigin, kTRUE, kFALSE, 0);
   else
      m_embeddedViewer->SetGuideState(TGLUtil::kAxesNone, kTRUE, kFALSE, 0);
}


void
FWGlimpseView::showCylinder( )
{
   if ( m_showCylinder.value() )
      m_cylinder->SetRnrState(kTRUE);
   else
      m_cylinder->SetRnrState(kFALSE);

   gEve->Redraw3D();
}

//
// static member functions
//
const std::string&
FWGlimpseView::staticTypeName()
{
   static std::string s_name("Glimpse");
   return s_name;
}


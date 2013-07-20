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
// $Id: FWGlimpseView.cc,v 1.42 2011/02/22 18:37:31 amraktad Exp $
//

#include <boost/bind.hpp>


#include "TGLPerspectiveCamera.h"
#include "TGLViewer.h"
#include "TGLScenePad.h"
#include "TEveScene.h"
#include "TEveViewer.h"

#include "TEveManager.h"
#include "TEveElement.h"

#include "TEveText.h"
#include "TGLFontManager.h"

#include "TEveTrans.h"
#include "TGeoTube.h"
#include "TEveGeoNode.h"
#include "TEveStraightLineSet.h"

// user include files
#include "Fireworks/Core/interface/FWGlimpseView.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//
//double FWGlimpseView::m_scale = 1;

//
// constructors and destructorquery
//
FWGlimpseView::FWGlimpseView(TEveWindowSlot* iParent, FWViewType::EType typeId) :
   FWEveView(iParent, typeId),
   m_cylinder(0),
   m_showAxes(this, "Show Axes", true ),
   m_showCylinder(this, "Show Cylinder", true)
{
   createAxis();

   // made new wireframe scene
   TEveScene* wns = gEve->SpawnNewScene(Form("Wireframe Scene %s", typeName().c_str()));
   viewer()->AddScene(wns);
   TGLScene* gls  = wns->GetGLScene();
   gls->SetStyle(TGLRnrCtx::kWireFrame);
   gls->SetLOD(TGLRnrCtx::kLODMed);
   gls->SetSelectable(kFALSE);

   TEveGeoManagerHolder gmgr(TEveGeoShape::GetGeoMangeur());
   TGeoTube* tube = new TGeoTube(129,130,310);
   m_cylinder = fireworks::getShape("Detector outline", tube, kWhite);
   m_cylinder->SetPickable(kFALSE);
   m_cylinder->SetMainColor(kGray+3);
   wns->AddElement(m_cylinder);

   TGLViewer* ev = viewerGL();
   ev->SetCurrentCamera(TGLViewer::kCameraPerspXOZ);
   m_showAxes.changed_.connect(boost::bind(&FWGlimpseView::showAxes,this));
   m_showCylinder.changed_.connect(boost::bind(&FWGlimpseView::showCylinder,this));
}

FWGlimpseView::~FWGlimpseView()
{
}


//
// member functions
//

void
FWGlimpseView::createAxis()
{
   // create 3D axes
   TEveElementList* axisHolder = new TEveElementList("GlimpseAxisHolder");

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
   axisHolder->AddElement(xAxis);

   TEveText* xTxt = new TEveText( "X+" );
   xTxt->PtrMainTrans()->SetPos(100-fs, -fs, 0);
   xTxt->SetFontMode(fontMode);
   xTxt->SetMainColor(fcol);
   axisHolder->AddElement(xTxt);

   // Y axis
   TEveStraightLineSet* yAxis = new TEveStraightLineSet( "GlimpseYAxis" );
   yAxis->SetPickable(kTRUE);
   yAxis->SetTitle("Energy Scale, 100 GeV, Y-axis (upward)");
   yAxis->SetLineColor(fcol);
   yAxis->SetLineStyle(3);
   yAxis->AddLine(0,-100,0,0,100,0);
   axisHolder->AddElement(yAxis);

   TEveText* yTxt = new TEveText( "Y+" );
   yTxt->PtrMainTrans()->SetPos(0, 100-fs, 0);
   yTxt->SetFontMode(fontMode);
   yTxt->SetMainColor(fcol);
   axisHolder->AddElement(yTxt);

   // Z axis
   TEveStraightLineSet* zAxis = new TEveStraightLineSet( "GlimpseZAxis" );
   zAxis->SetPickable(kTRUE);
   zAxis->SetTitle("Energy Scale, 100 GeV, Z-axis (west, along beam)");
   zAxis->SetLineColor(fcol);
   zAxis->AddLine(0,0,-100,0,0,100);
   axisHolder->AddElement(zAxis);

   TEveText* zTxt = new TEveText( "Z+" );
   zTxt->PtrMainTrans()->SetPos(0, -fs,  100 - zTxt->GetExtrude()*2);
   zTxt->SetFontMode(fontMode);
   zTxt->SetMainColor(fcol);
   axisHolder->AddElement(zTxt);

   geoScene()->AddElement(axisHolder);
}


void
FWGlimpseView::showAxes( )
{
   if ( m_showAxes.value() )
      viewerGL()->SetGuideState(TGLUtil::kAxesOrigin, kTRUE, kFALSE, 0);
   else
      viewerGL()->SetGuideState(TGLUtil::kAxesNone, kTRUE, kFALSE, 0);
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


void
FWGlimpseView::addTo(FWConfiguration& iTo) const
{
   FWEveView::addTo(iTo);   
   TGLPerspectiveCamera* camera = dynamic_cast<TGLPerspectiveCamera*>(&(viewerGL()->CurrentCamera()));
   if (camera)
      addToPerspectiveCamera(camera, typeName(), iTo);
}

void
FWGlimpseView::setFrom(const FWConfiguration& iFrom)
{
   FWEveView::setFrom(iFrom);
   TGLPerspectiveCamera* camera = dynamic_cast<TGLPerspectiveCamera*>(&(viewerGL()->CurrentCamera()));
   if (camera)
      setFromPerspectiveCamera(camera, typeName(), iFrom);
}


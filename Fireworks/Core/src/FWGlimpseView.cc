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
// $Id: FWGlimpseView.cc,v 1.36 2009/12/14 16:10:28 amraktad Exp $
//

#include <boost/bind.hpp>

// FIXME
// need camera parameters
#define private public
#include "TGLPerspectiveCamera.h"
#undef private


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
   FWEveView(iParent),
   m_cylinder(0),
   m_cameraMatrix(0),
   m_cameraMatrixBase(0),
   m_cameraFOV(0),
   m_showAxes(this, "Show Axes", true ),
   m_showCylinder(this, "Show Cylinder", true),
   m_scaler(iScaler)
{
   scene()->SetElementName(staticTypeName().c_str());
   scene()->AddElement(list);
   viewer()->SetElementName(staticTypeName().c_str());

   TGLViewer* ev = viewerGL();
   ev->SetCurrentCamera(TGLViewer::kCameraPerspXOZ);
   m_cameraMatrix = const_cast<TGLMatrix*>(&(ev->CurrentCamera().GetCamTrans()));
   m_cameraMatrixBase = const_cast<TGLMatrix*>(&(ev->CurrentCamera().GetCamBase()));
   if ( TGLPerspectiveCamera* camera =
        dynamic_cast<TGLPerspectiveCamera*>(&(ev->CurrentCamera())) )
      m_cameraFOV = &(camera->fFOV);


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
   scene()->AddElement(xAxis);

   TEveText* xTxt = new TEveText( "X+" );
   xTxt->PtrMainTrans()->SetPos(100-fs, -fs, 0);
   xTxt->SetFontMode(fontMode);
   xTxt->SetMainColor(fcol);
   scene()->AddElement(xTxt);

   // Y axis
   TEveStraightLineSet* yAxis = new TEveStraightLineSet( "GlimpseYAxis" );
   yAxis->SetPickable(kTRUE);
   yAxis->SetTitle("Energy Scale, 100 GeV, Y-axis (upward)");
   yAxis->SetLineColor(fcol);
   yAxis->SetLineStyle(3);
   yAxis->AddLine(0,-100,0,0,100,0);
   scene()->AddElement(yAxis);

   TEveText* yTxt = new TEveText( "Y+" );
   yTxt->PtrMainTrans()->SetPos(0, 100-fs, 0);
   yTxt->SetFontMode(fontMode);
   yTxt->SetMainColor(fcol);
   scene()->AddElement(yTxt);

   // Z axis
   TEveStraightLineSet* zAxis = new TEveStraightLineSet( "GlimpseZAxis" );
   zAxis->SetPickable(kTRUE);
   zAxis->SetTitle("Energy Scale, 100 GeV, Z-axis (west, along beam)");
   zAxis->SetLineColor(fcol);
   zAxis->AddLine(0,0,-100,0,0,100);
   scene()->AddElement(zAxis);

   TEveText* zTxt = new TEveText( "Z+" );
   zTxt->PtrMainTrans()->SetPos(0, -fs,  100 - zTxt->GetExtrude()*2);
   zTxt->SetFontMode(fontMode);
   zTxt->SetMainColor(fcol);
   scene()->AddElement(zTxt);

   // made detector outline in wireframe scene
   TEveScene* wns = gEve->SpawnNewScene(Form("Wireframe %s", staticTypeName().c_str()));
   viewer()->AddScene(wns);
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

   m_showAxes.changed_.connect(boost::bind(&FWGlimpseView::showAxes,this));
   m_showCylinder.changed_.connect(boost::bind(&FWGlimpseView::showCylinder,this));
}

FWGlimpseView::~FWGlimpseView()
{
}

void
FWGlimpseView::setFrom(const FWConfiguration& iFrom)
{
   // take care of parameters
   FWEveView::setFrom(iFrom);

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
   viewerGL()->RequestDraw();
}

const std::string&
FWGlimpseView::typeName() const
{
   return staticTypeName();
}

void
FWGlimpseView::addTo(FWConfiguration& iTo) const
{
   FWEveView::addTo(iTo);

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
FWGlimpseView::updateScale( double scale )
{
   m_scaler->setScale(scale);
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

//
// static member functions
//
const std::string&
FWGlimpseView::staticTypeName()
{
   static std::string s_name("Glimpse");
   return s_name;
}


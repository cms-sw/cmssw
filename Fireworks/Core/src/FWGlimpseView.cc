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
// $Id: FWGlimpseView.cc,v 1.9 2008/07/17 10:11:32 dmytro Exp $
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
#include "TEveElement.h"
#include "TEveCalo.h"
#include "TEveElement.h"
#include "TEveRGBAPalette.h"
#include "TEveLegoEventHandler.h"
#include "TGLWidget.h"
#include "TEveTrans.h"
#include "TGeoTube.h"
#include "TEveGeoNode.h"
#include "TEveGeoShapeExtract.h"
#include "TEveStraightLineSet.h"
#include "TGeoArb8.h"

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
FWGlimpseView::FWGlimpseView(TGFrame* iParent, TEveElementList* list, 
                             FWEveValueScaler* iScaler):
 m_cameraMatrix(0),
 m_cameraMatrixBase(0),
 m_cameraFOV(0),
 // m_scaleParam(this,"Energy scale", static_cast<double>(iScaler->scale()), 0.01, 1000.),
 m_showAxes(this, "Show Axes", true ),
 m_scaler(iScaler)
{
   m_pad = new TEvePad;
   TGLEmbeddedViewer* ev = new TGLEmbeddedViewer(iParent, m_pad);
   m_embeddedViewer=ev;
   TEveViewer* nv = new TEveViewer(staticTypeName().c_str());
   nv->SetGLViewer(ev);
   nv->IncDenyDestroy();
   // ev->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
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
   
   /*
   TEveStraightLineSet* xAxis = new TEveStraightLineSet( "GlimpseXAxis" );
   xAxis->SetPickable(kTRUE);
   xAxis->SetTitle("Energy Scale, 100 GeV, X-axis (LHC center)");
   xAxis->IncDenyDestroy();
   xAxis->SetLineColor(kGray);
   xAxis->AddLine(0,0,0,100,0,0);
   gEve->AddElement(xAxis, ns);

   TEveStraightLineSet* yAxis = new TEveStraightLineSet( "GlimpseXAxis" );
   yAxis->SetPickable(kTRUE);
   yAxis->SetTitle("Energy Scale, 100 GeV, Y-axis (upward)");
   yAxis->IncDenyDestroy();
   yAxis->SetLineColor(kGray);
   yAxis->AddLine(0,0,0,0,100,0);
   gEve->AddElement(yAxis, ns);

   TEveStraightLineSet* zAxis = new TEveStraightLineSet( "GlimpseXAxis" );
   zAxis->SetPickable(kTRUE);
   zAxis->SetTitle("Energy Scale, 100 GeV, Z-axis (west, along beam)");
   zAxis->IncDenyDestroy();
   zAxis->SetLineColor(kGray);
   zAxis->AddLine(0,0,0,0,0,100);
   gEve->AddElement(zAxis, ns);
   */
   /*
   // made detector outline
   TGeoTube* tube = new TGeoTube(129,130,310);
   TEveGeoShapeExtract* extract = fw::getShapeExtract("Detector outline", tube, kWhite);
   TEveElement* element = TEveGeoShape::ImportShapeExtract(extract, ns);
   element->SetPickable(kFALSE);
   element->SetMainTransparency(98);
   gEve->AddElement(element, ns);
   */
   
   /*
   TGeoTrap* cube = new TGeoTrap(100,0,0,100,100,100,0,100,100,100,0);
   TEveGeoShapeExtract* extract = fw::getShapeExtract("Detector outline", cube, Color_t(TColor::GetColor("#202020")) );
   TEveElement* element = TEveGeoShape::ImportShapeExtract(extract, ns);
   element->SetPickable(kFALSE);
   element->SetMainTransparency(50);
   gEve->AddElement(element, ns);
   */
   
   /*
   TEveStraightLineSet* outline = new TEveStraightLineSet( "EnergyScale" );
   outline->SetPickable(kTRUE);
   outline->SetTitle("100 GeV Energy Scale Cube");
   outline->IncDenyDestroy();
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
}

FWGlimpseView::~FWGlimpseView()
{
   //NOTE: have to do this EVIL activity to avoid double deletion. The fFrame inside glviewer
   // was added to a CompositeFrame which will delete it.  However, TGLEmbeddedViewer will also
   // delete fFrame in its destructor
   TGLEmbeddedViewer* glviewer = dynamic_cast<TGLEmbeddedViewer*>(m_viewer->GetGLViewer());
   glviewer->fFrame=0;
   delete glviewer;
   
   delete m_viewer;
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

     { 
	assert ( m_cameraFOV );	
	const FWConfiguration* value = iFrom.valueForKey( "Glimpse FOV" );
	assert( value );
	std::istringstream s(value->value());
	s>>*m_cameraFOV;
     }
   m_viewer->GetGLViewer()->RequestDraw();
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
      iTo.addKeyValue(matrixName+osIndex.str()+"Lego",FWConfiguration(osValue.str()));
   }
   
   // transformation matrix base
   assert(m_cameraMatrixBase);
   matrixName = "cameraMatrixBase";
   for ( unsigned int i = 0; i < 16; ++i ){
      std::ostringstream osIndex;
      osIndex << i;
      std::ostringstream osValue;
      osValue << (*m_cameraMatrixBase)[i];
      iTo.addKeyValue(matrixName+osIndex.str()+"Lego",FWConfiguration(osValue.str()));
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

//
// static member functions
//
const std::string& 
FWGlimpseView::staticTypeName()
{
   static std::string s_name("Glimpse");
   return s_name;
}


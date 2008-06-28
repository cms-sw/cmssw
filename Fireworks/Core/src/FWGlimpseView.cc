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
// $Id: FWGlimpseView.cc,v 1.2 2008/06/26 00:32:48 dmytro Exp $
//

// system include files
#include <algorithm>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/numeric/conversion/converter.hpp>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "TRootEmbeddedCanvas.h"
#include "THStack.h"
#include "TCanvas.h"
#include "TClass.h"
#include "TH2F.h"
#include "TView.h"
#include "TColor.h"
#include "TEveScene.h"
#include "TGLViewer.h"
#include "TGLEmbeddedViewer.h"
#include "TEveViewer.h"
#include "TEveManager.h"
#include "TEveElement.h"
#include "TEveCalo.h"
#include "TEveElement.h"
#include "TEveRGBAPalette.h"
#include "TGLPerspectiveCamera.h"
#include "TEveLegoEventHandler.h"
#include "TGLWidget.h"
#include "TEveTrans.h"
#include "TGeoTube.h"
#include "TEveGeoNode.h"
#include "TEveGeoShapeExtract.h"

// user include files
#include "Fireworks/Core/interface/FWGlimpseView.h"
#include "Fireworks/Core/interface/FWGlimpseViewManager.h"
#include "Fireworks/Core/interface/FWConfiguration.h"
#include "Fireworks/Core/interface/BuilderUtils.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//
double FWGlimpseView::m_scale = 1;

//
// constructors and destructor
//
FWGlimpseView::FWGlimpseView(TGFrame* iParent, TEveElementList* list):
 m_cameraMatrix(0),
 m_cameraMatrixBase(0),
 m_scaleParam(this,"Energy scale", 2.0, 0.01, 1000.),
 m_manager(0)
{
   m_pad = new TEvePad;
   TGLEmbeddedViewer* ev = new TGLEmbeddedViewer(iParent, m_pad);
   m_embeddedViewer=ev;
   ev->SetGuideState(TGLUtil::kAxesOrigin, kTRUE, kFALSE, 0);
   TEveViewer* nv = new TEveViewer(staticTypeName().c_str());
   nv->SetGLViewer(ev);
   nv->IncDenyDestroy();
   // ev->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
   ev->SetCurrentCamera(TGLViewer::kCameraPerspXOZ);
   //? ev->SetEventHandler(new TGlimpseEventHandler("Lego", ev->GetGLWidget(), ev));
   m_cameraMatrix = const_cast<TGLMatrix*>(&(ev->CurrentCamera().GetCamTrans()));
   m_cameraMatrixBase = const_cast<TGLMatrix*>(&(ev->CurrentCamera().GetCamBase()));

   TEveScene* ns = gEve->SpawnNewScene(staticTypeName().c_str());
   m_scene = ns;
   nv->AddScene(ns);
   m_viewer=nv;
   gEve->AddElement(nv, gEve->GetViewers());
   gEve->AddElement(list,ns);
   gEve->AddToListTree(list, kTRUE);
   
   // made detector outline
   TGeoTube* tube = new TGeoTube(129,130,310);
   TEveGeoShapeExtract* extract = fw::getShapeExtract("Detector outline", tube, kWhite);
   TEveElement* element = TEveGeoShape::ImportShapeExtract(extract, ns);
   element->SetPickable(kFALSE);
   element->SetMainTransparency(98);
   gEve->AddElement(element, ns);
   
   m_scaleParam.changed_.connect(boost::bind(&FWGlimpseView::updateScale,this,_1));
}

FWGlimpseView::~FWGlimpseView()
{
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
FWGlimpseView::setScale( double scale ) 
{ 
   m_scale = scale; 
}

double 
FWGlimpseView::getScale() 
{ 
   return m_scale; 
}

void 
FWGlimpseView::setManager( FWGlimpseViewManager* manager ) 
{ 
   m_manager = manager; 
}

void 
FWGlimpseView::updateScale( double scale ) 
{ 
   setScale( scale );
   if ( m_manager ) m_manager->newEventAvailable();
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


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
// $Id: FWEveLegoView.cc,v 1.4 2008/03/27 11:05:17 dmytro Exp $
//

// system include files
#include <algorithm>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/numeric/conversion/converter.hpp>
#include <iostream>
#include <sstream>

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
FWEveLegoView::FWEveLegoView(TGFrame* iParent, TEveElementList* list):
 m_range(this,"energy threshold (%)",0.,0.,100.),
 m_cameraMatrix(0),
 m_cameraMatrixBase(0)
{
   m_pad = new TEvePad;
   TGLEmbeddedViewer* ev = new TGLEmbeddedViewer(iParent, m_pad);
   m_embeddedViewer=ev;
   TEveViewer* nv = new TEveViewer(staticTypeName().c_str());
   nv->SetGLViewer(ev);
   nv->IncDenyDestroy();
   // ev->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
   ev->SetCurrentCamera(TGLViewer::kCameraPerspXOY);
   m_cameraMatrix = const_cast<TGLMatrix*>(&(ev->CurrentCamera().GetCamTrans()));
   m_cameraMatrixBase = const_cast<TGLMatrix*>(&(ev->CurrentCamera().GetCamBase()));

   TEveScene* ns = gEve->SpawnNewScene(staticTypeName().c_str());
   m_scene = ns;
   nv->AddScene(ns);
   m_viewer=nv;
   gEve->AddElement(nv, gEve->GetViewers());
   
   TEveRGBAPalette* pal = new TEveRGBAPalette(0, 100);
   // pal->SetLimits(0, data->GetMaxVal());
   pal->SetLimits(0, 100);
   pal->SetDefaultColor((Color_t)1000);
   
   m_lego = new TEveCaloLego();
   m_lego->SetPalette(pal);
   m_lego->SetMainColor(Color_t(TColor::GetColor("#0A0A0A")));
   // lego->SetEtaLimits(etaLimLow, etaLimHigh);
   // lego->SetTitle("caloTower Et distribution");
   gEve->AddElement(m_lego, ns);
   gEve->AddToListTree(m_lego, kTRUE);
   gEve->AddElement(list,ns);
   gEve->AddToListTree(list, kTRUE);
   m_range.changed_.connect(boost::bind(&FWEveLegoView::doMinThreshold,this,_1));
}

FWEveLegoView::~FWEveLegoView()
{
}

void
FWEveLegoView::draw(TEveCaloDataHist* data)
{
   // bool firstTime = (m_lego->GetData() == 0);
   m_lego->SetData(data);
   m_lego->ElementChanged();
   m_lego->InvalidateCache();
   /*
   if ( firstTime ) {
      m_scene->Repaint();
      m_viewer->Redraw(kTRUE);
      m_viewer->GetGLViewer()->ResetCurrentCamera();
   }
   */ 
   m_viewer->GetGLViewer()->RequestDraw();
}

void 
FWEveLegoView::doMinThreshold(double value)
{
   m_lego->GetPalette()->SetMin( int(value) );
   m_lego->ElementChanged();
   m_lego->InvalidateCache();
   m_viewer->GetGLViewer()->RequestDraw();
}

void 
FWEveLegoView::setFrom(const FWConfiguration& iFrom)
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


//
// static member functions
//
const std::string& 
FWEveLegoView::staticTypeName()
{
   static std::string s_name("3D Lego Pro");
   return s_name;
}


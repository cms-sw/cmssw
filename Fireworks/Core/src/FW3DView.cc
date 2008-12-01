// -*- C++ -*-
//
// Package:     Core
// Class  :     FW3DView
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 21 11:22:41 EST 2008
// $Id: FW3DView.cc,v 1.20 2008/11/10 18:07:57 amraktad Exp $
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
#include "TGLScenePad.h"
#include "TGLFontManager.h"
#include "TEveTrans.h"
#include "TGeoTube.h"
#include "TEveGeoNode.h"
#include "TEveStraightLineSet.h"
#include "TEveText.h"
#include "TGeoArb8.h"

// user include files
#include "Fireworks/Core/interface/FW3DView.h"
#include "Fireworks/Core/interface/FWEveValueScaler.h"
#include "Fireworks/Core/interface/FWConfiguration.h"
#include "Fireworks/Core/interface/BuilderUtils.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//
//double FW3DView::m_scale = 1;

//
// constructors and destructor
//
FW3DView::FW3DView(TGFrame* iParent, TEveElementList* list):
 m_cameraMatrix(0),
 m_cameraMatrixBase(0),
 m_cameraFOV(0)
{
   m_pad = new TEvePad;
   TGLEmbeddedViewer* ev = new TGLEmbeddedViewer(iParent, m_pad, 0);
   m_embeddedViewer=ev;
   TEveViewer* nv = new TEveViewer(staticTypeName().c_str());
   nv->SetGLViewer(ev);
   ev->SetCurrentCamera(TGLViewer::kCameraPerspXOZ);
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
}

FW3DView::~FW3DView()
{
   //NOTE: have to do this EVIL activity to avoid double deletion. The fFrame inside glviewer
   // was added to a CompositeFrame which will delete it.  However, TGLEmbeddedViewer will also
   // delete fFrame in its destructor
   TGLEmbeddedViewer* glviewer = dynamic_cast<TGLEmbeddedViewer*>(m_viewer->GetGLViewer());
   glviewer->fFrame=0;
   delete glviewer;

   m_viewer->Destroy();
   m_scene->Destroy();
   //delete m_viewer;
}

void
FW3DView::setFrom(const FWConfiguration& iFrom)
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
      const FWConfiguration* value = iFrom.valueForKey( matrixName + os.str() + "Plain3D" );
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
      const FWConfiguration* value = iFrom.valueForKey( matrixName + os.str() + "Plain3D" );
      assert( value );
      std::istringstream s(value->value());
      s>>((*m_cameraMatrixBase)[i]);
   }

     {
	assert ( m_cameraFOV );
	const FWConfiguration* value = iFrom.valueForKey( "Plain3D FOV" );
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
FW3DView::frame() const
{
   return m_embeddedViewer->GetFrame();
}

const std::string&
FW3DView::typeName() const
{
   return staticTypeName();
}

void
FW3DView::addTo(FWConfiguration& iTo) const
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
      iTo.addKeyValue(matrixName+osIndex.str()+"Plain3D",FWConfiguration(osValue.str()));
   }

   // transformation matrix base
   assert(m_cameraMatrixBase);
   matrixName = "cameraMatrixBase";
   for ( unsigned int i = 0; i < 16; ++i ){
      std::ostringstream osIndex;
      osIndex << i;
      std::ostringstream osValue;
      osValue << (*m_cameraMatrixBase)[i];
      iTo.addKeyValue(matrixName+osIndex.str()+"Plain3D",FWConfiguration(osValue.str()));
   }
     {
	assert ( m_cameraFOV );
	std::ostringstream osValue;
	osValue << *m_cameraFOV;
	iTo.addKeyValue("Plain3D FOV",FWConfiguration(osValue.str()));
     }
}

void
FW3DView::saveImageTo(const std::string& iName) const
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
FW3DView::staticTypeName()
{
   static std::string s_name("3D");
   return s_name;
}


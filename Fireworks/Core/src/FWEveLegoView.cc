
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
// $Id: FWEveLegoView.cc,v 1.71 2010/03/16 11:51:54 amraktad Exp $
//

// system include files
#include <boost/bind.hpp>

#include "TAttAxis.h"

#define private public
#include "TGLOrthoCamera.h"
#undef private
#include "TGLViewer.h"
#include "TGLPerspectiveCamera.h"
#include "TGLWidget.h"

#include "TEveManager.h"
#include "TEveElement.h"
#include "TEveScene.h"
#include "TEveViewer.h"
#include "TEveCalo.h"
#include "TEveTrans.h"
#include "TEveStraightLineSet.h"
#include "TEveCaloLegoOverlay.h"

// user include files
#include "Fireworks/Core/interface/FWGLEventHandler.h"
#include "Fireworks/Core/interface/FWConfiguration.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "Fireworks/Core/interface/FWEveLegoView.h"


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
   FWEveView(iParent),
   m_lego(0),
   m_overlay(0),
   m_plotEt(this,"Plot Et",true),   
   m_autoRebin(this,"Auto rebin on zoom",true),
   m_pixelsPerBin(this, "Pixels per bin", 10., 1., 20.),
   m_showScales(this,"Show scales", true),
   m_legoFixedScale(this,"Lego scale GeV)",100.,1.,1000.),
   m_legoAutoScale (this,"Lego auto scale",true),
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
   scene()->SetElementName(staticTypeName().c_str());
   scene()->AddElement(list);
   viewer()->SetElementName(staticTypeName().c_str());

   FWGLEventHandler* eh = new FWGLEventHandler((TGWindow*)viewerGL()->GetGLWidget(), (TObject*)viewerGL());
   viewerGL()->SetEventHandler(eh);


   if (list->HasChildren())
   {
      m_lego =  dynamic_cast<TEveCaloLego*>( list->FirstChild());
      if (m_lego) {
         m_overlay = new TEveCaloLegoOverlay();
         m_overlay->SetShowPlane(kFALSE);
         m_overlay->SetShowPerspective(kFALSE);
         m_overlay->GetAttAxis()->SetLabelSize(0.02);
         viewerGL()->AddOverlayElement(m_overlay);
         m_overlay->SetCaloLego(m_lego);
         m_overlay->SetShowScales(1); //temporary
         m_overlay->SetScalePosition(0.8, 0.6);
         m_overlay->SetScaleColorTransparency(kWhite, 0);

         viewerGL()->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
         eh->SetLego(m_lego);
      }
   }
   // take care of cameras
   //
   if ( TGLPerspectiveCamera* camera = dynamic_cast<TGLPerspectiveCamera*>( &(viewerGL()->RefCamera(TGLViewer::kCameraPerspXOY) ))) {
      m_cameraMatrixRef = const_cast<TGLMatrix*>(&(camera->GetCamTrans()));
      m_cameraMatrixBaseRef = const_cast<TGLMatrix*>(&(camera->GetCamBase()));
   }
   if ( TGLOrthoCamera* camera = dynamic_cast<TGLOrthoCamera*>( &(viewerGL()->RefCamera(TGLViewer::kCameraOrthoXOY) ))) {
      m_orthoCameraZoomRef = &(camera->fZoom);
      m_orthoCameraMatrixRef = const_cast<TGLMatrix*>(&(camera->GetCamTrans()));
   }

   m_autoRebin.changed_.connect(boost::bind(&FWEveLegoView::setAutoRebin,this));
   m_pixelsPerBin.changed_.connect(boost::bind(&FWEveLegoView::setPixelsPerBin,this));
   m_plotEt.changed_.connect(boost::bind(&FWEveLegoView::plotEt,this));
   m_showScales.changed_.connect(boost::bind(&FWEveLegoView::showScales,this));
   m_legoFixedScale.changed_.connect(boost::bind(&FWEveLegoView::updateLegoScale, this));
   m_legoAutoScale .changed_.connect(boost::bind(&FWEveLegoView::updateLegoScale, this));
}

FWEveLegoView::~FWEveLegoView()
{
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
   scene()->Repaint();
   viewer()->Redraw(kTRUE);

   if ( m_cameraMatrix && m_cameraMatrixBase && m_orthoCameraMatrix) {
      *m_cameraMatrixRef = *m_cameraMatrix;
      *m_cameraMatrixBaseRef = *m_cameraMatrixBase;
      *m_orthoCameraMatrixRef = *m_orthoCameraMatrix;
      *m_orthoCameraZoomRef = m_orthoCameraZoom;
   }
   if (!m_topView ) {
      viewerGL()->SetCurrentCamera(TGLViewer::kCameraPerspXOY);
   }

   m_cameraSet = true;
   viewerGL()->RequestDraw();
}

void
FWEveLegoView::setFrom(const FWConfiguration& iFrom)
{
   // take care of parameters
   FWEveView::setFrom(iFrom);

   // retrieve camera parameters
   m_cameraMatrix = new TGLMatrix();
   m_cameraMatrixBase = new TGLMatrix();
   m_orthoCameraMatrix = new TGLMatrix();

   // transformation matrix
   assert(m_cameraMatrix);
   std::string matrixName("cameraMatrix");
   for ( unsigned int i = 0; i < 16; ++i ) {
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
   for ( unsigned int i = 0; i < 16; ++i ) {
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
   for ( unsigned int i = 0; i < 16; ++i ) {
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

void
FWEveLegoView::setPixelsPerBin()
{
   if(m_lego) {
      m_lego->SetPixelsPerBin((Int_t) (m_pixelsPerBin.value()));
      m_lego->ElementChanged(kTRUE,kTRUE);
   }
}

void
FWEveLegoView::plotEt()
{
   if (m_lego)
   {
      m_lego->SetPlotEt(m_plotEt.value());
      viewerGL()->RequestDraw();
   }
}

void
FWEveLegoView::showScales()
{
   m_overlay->SetShowScales(m_showScales.value());
   viewerGL()->RequestDraw();
}

void
FWEveLegoView::updateLegoScale()
{
  if (m_lego)
   {
      m_lego->SetMaxValAbs( m_legoFixedScale.value() );
      m_lego->SetScaleAbs ( ! m_legoAutoScale.value() );
      m_lego->ElementChanged(kTRUE,kTRUE);
      //m_lego->ElementChanged();
      //gEve->Redraw3D();
   } 
}

//
// const member functions
//

const std::string&
FWEveLegoView::typeName() const
{
   return staticTypeName();
}

void
FWEveLegoView::addTo(FWConfiguration& iTo) const
{
   FWEveView::addTo(iTo);

   // store camera parameters

   // transformation matrix
   assert(m_cameraMatrixRef);
   std::string matrixName("cameraMatrix");
   for ( unsigned int i = 0; i < 16; ++i ) {
      std::ostringstream osIndex;
      osIndex << i;
      std::ostringstream osValue;
      osValue << (*m_cameraMatrixRef)[i];
      iTo.addKeyValue(matrixName+osIndex.str()+"Lego",FWConfiguration(osValue.str()));
   }

   // transformation matrix base
   assert(m_cameraMatrixBaseRef);
   matrixName = "cameraMatrixBase";
   for ( unsigned int i = 0; i < 16; ++i ) {
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
   if ( dynamic_cast<TGLOrthoCamera*>( &(viewerGL()->CurrentCamera()) ) )
      topView = true;
   s << topView;
   name = "topView";
   iTo.addKeyValue(name+typeName(),FWConfiguration(s.str()));

   // transformation matrix
   assert(m_orthoCameraMatrixRef);
   std::string orthoMatrixName("orthoCameraMatrix");
   for ( unsigned int i = 0; i < 16; ++i ) {
      std::ostringstream osIndex;
      osIndex << i;
      std::ostringstream osValue;
      osValue << (*m_orthoCameraMatrixRef)[i];
      iTo.addKeyValue(orthoMatrixName+osIndex.str()+typeName(),FWConfiguration(osValue.str()));
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


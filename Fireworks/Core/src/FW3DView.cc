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
// $Id: FW3DView.cc,v 1.2 2008/12/02 09:01:51 dmytro Exp $
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

#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "TEveGeoNode.h"
#include "TEveScene.h"
#include "Fireworks/Core/interface/TEveElementIter.h"
#include "TEvePolygonSetProjected.h"

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
 m_cameraFOV(0),
 m_muonBarrelElements(0),
 m_muonEndcapElements(0),
 m_showMuonBarrel(this, "Show Muon Barrel", true ),
 m_showMuonEndcap(this, "Show Muon Endcap", true),
 m_showWireFrame(this, "Show Wire Frame", true),
 m_geomTransparency(this,"Detector Transparency", 95l, 0l, 100l)
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

   m_scene = gEve->SpawnNewScene(staticTypeName().c_str());
   nv->AddScene(m_scene);
   
   m_detectorScene = gEve->SpawnNewScene((staticTypeName()+"detector").c_str());
   nv->AddScene(m_detectorScene);
   
   m_viewer=nv;
   gEve->AddElement(nv, gEve->GetViewers());
   gEve->AddElement(list,m_scene);
   gEve->AddToListTree(list, kTRUE);
   m_showMuonBarrel.changed_.connect(boost::bind(&FW3DView::showMuonBarrel,this));
   m_showMuonEndcap.changed_.connect(boost::bind(&FW3DView::showMuonEndcap,this));
   m_showWireFrame.changed_.connect(boost::bind(&FW3DView::showWireFrame,this));
   m_geomTransparency.changed_.connect(boost::bind(&FW3DView::setTransparency,this));
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


void
FW3DView::showMuonBarrel( )
{
   if ( ! m_muonBarrelElements ) return; 
   if ( m_showMuonBarrel.value() )
     m_muonBarrelElements->SetRnrState(kTRUE);
   else
     m_muonBarrelElements->SetRnrState(kFALSE);
   gEve->Redraw3D();
}

void
FW3DView::showMuonEndcap( )
{
   if ( ! m_muonEndcapElements ) return; 
   if ( m_showMuonEndcap.value() )
     m_muonEndcapElements->SetRnrState(kTRUE);
   else
     m_muonEndcapElements->SetRnrState(kFALSE);
   gEve->Redraw3D();
}

void
FW3DView::showWireFrame( )
{
   if ( m_showWireFrame.value() )
     m_detectorScene->GetGLScene()->SetStyle(TGLRnrCtx::kWireFrame);
   else
     m_detectorScene->GetGLScene()->SetStyle(TGLRnrCtx::kFill);
   m_embeddedViewer->RequestDraw(TGLRnrCtx::kLODHigh);
   // gEve->GetViewers()->RepaintAllViewers(kFALSE, kFALSE);
   // gEve->Redraw3D();
}

void
FW3DView::setTransparency( )
{
   if ( m_muonBarrelElements ) {
      TEveElementIter iter(m_muonBarrelElements);
      while ( TEveElement* element = iter.current() ) {
	 element->SetMainTransparency(m_geomTransparency.value());
	 /*
	 element->SetMainColor(Color_t(TColor::GetColor("#3f0000")));
	 if ( TEvePolygonSetProjected* poly = dynamic_cast<TEvePolygonSetProjected*>(element) )
	   poly->SetLineColor(Color_t(TColor::GetColor("#7f0000")));
	  */
	 iter.next();
      }
      gEve->Redraw3D();
   }
   if ( m_muonEndcapElements ) {
      TEveElementIter iter(m_muonEndcapElements);
      while ( TEveElement* element = iter.current() ) {
	 element->SetMainTransparency(m_geomTransparency.value());
	 /*
	 element->SetMainColor(Color_t(TColor::GetColor("#3f0000")));
	 if ( TEvePolygonSetProjected* poly = dynamic_cast<TEvePolygonSetProjected*>(element) )
	   poly->SetLineColor(Color_t(TColor::GetColor("#7f0000")));
	  */
	 iter.next();
      }
      gEve->Redraw3D();
   }
}


void FW3DView::makeGeometry( const DetIdToMatrix* geom )
{
   if ( ! geom ) {
      std::cout << "Warning: cannot get geometry to rendered detector outline. Skipped" << std::endl;
      return;
   }
   
   // barrel
   m_muonBarrelElements = new TEveElementList( "DT" );
   gEve->AddElement( m_muonBarrelElements, m_detectorScene );
   for ( Int_t iWheel = -2; iWheel <= 2; ++iWheel)
     for (Int_t iStation = 1; iStation <= 4; ++iStation)
       {
	  std::ostringstream s;
	  s << "Station" << iStation;
	  TEveElementList* cStation  = new TEveElementList( s.str().c_str() );
	  m_muonBarrelElements->AddElement( cStation );
	  for (Int_t iSector = 1 ; iSector <= 14; ++iSector)
	    {
	       if ( iStation < 4 && iSector > 12 ) continue;
	       DTChamberId id(iWheel, iStation, iSector);
	       TEveGeoShape* shape = geom->getShape( id.rawId() );
	       if ( !shape ) continue;
	       shape->SetMainTransparency(m_geomTransparency.value());
	       cStation->AddElement(shape);
	    }
       }
   
   m_muonEndcapElements = new TEveElementList( "CSC" );
   gEve->AddElement( m_muonEndcapElements, m_detectorScene );
   for ( Int_t iEndcap = 1; iEndcap <= 2; ++iEndcap ) {// 1=forward (+Z), 2=backward(-Z)
      TEveElementList* cEndcap = 0;
      if (iEndcap == 1)
	cEndcap = new TEveElementList( "Forward" );
      else
	cEndcap = new TEveElementList( "Backward" );
      m_muonEndcapElements->AddElement( cEndcap );
      for ( Int_t iStation=1; iStation<=4; ++iStation)
      {
	 std::ostringstream s; s << "Station" << iStation;
	 TEveElementList* cStation  = new TEveElementList( s.str().c_str() );
	 cEndcap->AddElement( cStation );
	 for ( Int_t iRing=1; iRing<=4; ++iRing) {
	    if (iStation > 1 && iRing > 2) continue;
	    std::ostringstream s; s << "Ring" << iRing;
	    TEveElementList* cRing  = new TEveElementList( s.str().c_str() );
	    cStation->AddElement( cRing );
	    for ( Int_t iChamber=1; iChamber<=72; ++iChamber)
            {
	       if (iStation>1 && iChamber>36) continue;
	       Int_t iLayer = 0; // chamber
	       // exception is thrown if parameters are not correct and I keep
	       // forgetting how many chambers we have in each ring.
	       try {
		  CSCDetId id(iEndcap, iStation, iRing, iChamber, iLayer);
		  TEveGeoShape* shape = geom->getShape( id.rawId() );
		  if ( ! shape ) continue;
		  shape->SetMainTransparency(m_geomTransparency.value());
		  cRing->AddElement( shape );
	       }
	       catch ( ... ) {}
	    }
	 }
      }
   }


/*
   // set background geometry visibility parameters

   TEveElementIter rhoPhiDT(m_rhoPhiGeomProjMgr.get(),"MuonRhoPhi");
   if ( rhoPhiDT.current() ) {
      m_rhoPhiGeom.push_back( rhoPhiDT.current() );
      TEveElementIter iter(rhoPhiDT.current());
      while ( TEveElement* element = iter.current() ) {
	 element->SetMainTransparency(50);
	 element->SetMainColor(Color_t(TColor::GetColor("#3f0000")));
	 if ( TEvePolygonSetProjected* poly = dynamic_cast<TEvePolygonSetProjected*>(element) )
	   poly->SetLineColor(Color_t(TColor::GetColor("#7f0000")));
	 iter.next();
      }
   }
*/
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

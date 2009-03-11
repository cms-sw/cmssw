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
// $Id: FW3DView.cc,v 1.11 2009/03/04 17:00:46 chrjones Exp $
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
#include "TEveWindow.h"
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
#include "PhysicsTools/Utilities/interface/deltaR.h"

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
FW3DView::FW3DView(TEveWindowSlot* iParent, TEveElementList* list) :
   m_cameraMatrix(0),
   m_cameraMatrixBase(0),
   m_cameraFOV(0),
   m_muonBarrelElements(0),
   m_muonEndcapElements(0),
   m_pixelBarrelElements(0),
   m_pixelEndcapElements(0),
   m_trackerBarrelElements(0),
   m_trackerEndcapElements(0),
   m_showMuonBarrel(this, "Show Muon Barrel", true ),
   m_showMuonEndcap(this, "Show Muon Endcap", true),
   m_showPixelBarrel(this, "Show Pixel Barrel", false ),
   m_showPixelEndcap(this, "Show Pixel Endcap", false),
   m_showTrackerBarrel(this, "Show Tracker Barrel", false ),
   m_showTrackerEndcap(this, "Show Tracker Endcap", false),
   m_showWireFrame(this, "Show Wire Frame", true),
   m_geomTransparency(this,"Detector Transparency", 95l, 0l, 100l)
{
   TEveViewer* nv = new TEveViewer(staticTypeName().c_str());
   m_embeddedViewer =  nv->SpawnGLEmbeddedViewer();
   iParent->ReplaceWindow(nv);

   TGLEmbeddedViewer* ev = m_embeddedViewer;
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
   m_detectorScene->GetGLScene()->SetSelectable(kFALSE);

   m_viewer=nv;
   gEve->AddElement(nv, gEve->GetViewers());
   gEve->AddElement(list,m_scene);
   gEve->AddToListTree(list, kTRUE);

   //make sure our defaults are honored
   showWireFrame();

   m_showMuonBarrel.changed_.connect(boost::bind(&FW3DView::showMuonBarrel,this));
   m_showMuonEndcap.changed_.connect(boost::bind(&FW3DView::showMuonEndcap,this));
   m_showPixelBarrel.changed_.connect(boost::bind(&FW3DView::showPixelBarrel,this));
   m_showPixelEndcap.changed_.connect(boost::bind(&FW3DView::showPixelEndcap,this));
   m_showTrackerBarrel.changed_.connect(boost::bind(&FW3DView::showTrackerBarrel,this));
   m_showTrackerEndcap.changed_.connect(boost::bind(&FW3DView::showTrackerEndcap,this));
   m_showWireFrame.changed_.connect(boost::bind(&FW3DView::showWireFrame,this));
   m_geomTransparency.changed_.connect(boost::bind(&FW3DView::setTransparency,this));
}

FW3DView::~FW3DView()
{
   m_scene->Destroy();
   m_viewer->DestroyWindowAndSlot();
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
   if ( !m_muonBarrelElements ) return;
   if ( m_showMuonBarrel.value() )
      m_muonBarrelElements->SetRnrState(kTRUE);
   else
      m_muonBarrelElements->SetRnrState(kFALSE);
   gEve->Redraw3D();
}

void
FW3DView::showMuonEndcap( )
{
   if ( !m_muonEndcapElements ) return;
   if ( m_showMuonEndcap.value() )
      m_muonEndcapElements->SetRnrState(kTRUE);
   else
      m_muonEndcapElements->SetRnrState(kFALSE);
   gEve->Redraw3D();
}

void
FW3DView::showPixelBarrel( )
{
   if ( !m_pixelBarrelElements ) return;
   if ( m_showPixelBarrel.value() )
      m_pixelBarrelElements->SetRnrState(kTRUE);
   else
      m_pixelBarrelElements->SetRnrState(kFALSE);
   gEve->Redraw3D();
}

void
FW3DView::showPixelEndcap( )
{
   if ( !m_pixelEndcapElements ) return;
   if ( m_showPixelEndcap.value() )
      m_pixelEndcapElements->SetRnrState(kTRUE);
   else
      m_pixelEndcapElements->SetRnrState(kFALSE);
   gEve->Redraw3D();
}

void
FW3DView::showTrackerBarrel( )
{
   if ( !m_trackerBarrelElements ) return;
   if ( m_showTrackerBarrel.value() )
      m_trackerBarrelElements->SetRnrState(kTRUE);
   else
      m_trackerBarrelElements->SetRnrState(kFALSE);
   gEve->Redraw3D();
}

void
FW3DView::showTrackerEndcap( )
{
   if ( !m_trackerEndcapElements ) return;
   if ( m_showTrackerEndcap.value() )
      m_trackerEndcapElements->SetRnrState(kTRUE);
   else
      m_trackerEndcapElements->SetRnrState(kFALSE);
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
   }
   if ( m_pixelBarrelElements ) {
      TEveElementIter iter(m_pixelBarrelElements);
      while ( TEveElement* element = iter.current() ) {
         element->SetMainTransparency(m_geomTransparency.value());
         iter.next();
      }
   }
   if ( m_pixelEndcapElements ) {
      TEveElementIter iter(m_pixelEndcapElements);
      while ( TEveElement* element = iter.current() ) {
         element->SetMainTransparency(m_geomTransparency.value());
         iter.next();
      }
   }
   if ( m_trackerBarrelElements ) {
      TEveElementIter iter(m_trackerBarrelElements);
      while ( TEveElement* element = iter.current() ) {
         /*
            if ( TEveTrans* matrix = element->PtrMainTrans() ) {
            // if ( TEveGeoShape* shape = dynamic_cast<TEveGeoShape*>(element) ) {
            // TVector3 shapeCenter( shape->GetShape()->GetTransform()->GetTranslation()[0],
            // shape->GetShape()->GetTransform()->GetTranslation()[1],
            // shape->GetShape()->GetTransform()->GetTranslation()[2] );
            TVector3 shapeCenter( matrix->GetBaseVec(4)[0],
                                matrix->GetBaseVec(4)[1],
                                matrix->GetBaseVec(4)[2] );
            double delta = reco::deltaR( 0.5, 3.14, shapeCenter.Eta(), shapeCenter.Phi() );
            if ( delta < 0.1 )
              element->SetMainTransparency(0);
            else
              element->SetMainTransparency(m_geomTransparency.value());
            } else
          */
         element->SetMainTransparency(m_geomTransparency.value());
         iter.next();
      }
   }
   if ( m_trackerEndcapElements ) {
      TEveElementIter iter(m_trackerEndcapElements);
      while ( TEveElement* element = iter.current() ) {
         element->SetMainTransparency(m_geomTransparency.value());
         iter.next();
      }
   }
   gEve->Redraw3D();
}


void FW3DView::makeGeometry( const DetIdToMatrix* geom )
{
   if ( !geom ) {
      std::cout << "Warning: cannot get geometry to rendered detector outline. Skipped" << std::endl;
      return;
   }

   // barrel muon
   m_muonBarrelElements = new TEveElementList( "DT" );
   m_muonBarrelElements->SetRnrState(m_showMuonBarrel.value());
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

   // endcap muon
   m_muonEndcapElements = new TEveElementList( "CSC" );
   m_muonEndcapElements->SetRnrState(m_showMuonEndcap.value());
   gEve->AddElement( m_muonEndcapElements, m_detectorScene );
   for ( Int_t iEndcap = 1; iEndcap <= 2; ++iEndcap ) { // 1=forward (+Z), 2=backward(-Z)
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
                  if ( !shape ) continue;
                  shape->SetMainTransparency(m_geomTransparency.value());
                  cRing->AddElement( shape );
               }
               catch (... ) {}
            }
         }
      }
   }

   // pixel barrel
   m_pixelBarrelElements = new TEveElementList( "PixelBarrel" );
   m_pixelBarrelElements->SetRnrState(m_showPixelBarrel.value());
   gEve->AddElement( m_pixelBarrelElements, m_detectorScene );
   std::vector<unsigned int> ids = geom->getMatchedIds("PixelBarrel");
   for ( std::vector<unsigned int>::const_iterator id = ids.begin();
         id != ids.end(); ++id ) {
      TEveGeoShape* shape = geom->getShape( *id );
      if ( !shape ) continue;
      shape->SetMainTransparency(m_geomTransparency.value());
      m_pixelBarrelElements->AddElement( shape );
   }

   // pixel endcap
   m_pixelEndcapElements = new TEveElementList( "PixelEndcap" );
   m_pixelEndcapElements->SetRnrState(m_showPixelEndcap.value());
   gEve->AddElement( m_pixelEndcapElements, m_detectorScene );
   ids = geom->getMatchedIds("PixelForward");
   for ( std::vector<unsigned int>::const_iterator id = ids.begin();
         id != ids.end(); ++id ) {
      TEveGeoShape* shape = geom->getShape( *id );
      if ( !shape ) continue;
      shape->SetMainTransparency(m_geomTransparency.value());
      m_pixelEndcapElements->AddElement( shape );
   }

   // tracker barrel
   m_trackerBarrelElements = new TEveElementList( "TrackerBarrel" );
   m_trackerBarrelElements->SetRnrState(m_showTrackerBarrel.value());
   gEve->AddElement( m_trackerBarrelElements, m_detectorScene );
   ids = geom->getMatchedIds("tib:TIB");
   for ( std::vector<unsigned int>::const_iterator id = ids.begin();
         id != ids.end(); ++id ) {
      TEveGeoShape* shape = geom->getShape( *id );
      if ( !shape ) continue;
      shape->SetMainTransparency(m_geomTransparency.value());
      m_trackerBarrelElements->AddElement( shape );
   }
   ids = geom->getMatchedIds("tob:TOB");
   for ( std::vector<unsigned int>::const_iterator id = ids.begin();
         id != ids.end(); ++id ) {
      TEveGeoShape* shape = geom->getShape( *id );
      if ( !shape ) continue;
      shape->SetMainTransparency(m_geomTransparency.value());
      m_trackerBarrelElements->AddElement( shape );
   }

   // tracker endcap
   m_trackerEndcapElements = new TEveElementList( "TrackerEndcap" );
   m_trackerEndcapElements->SetRnrState(m_showTrackerEndcap.value());
   gEve->AddElement( m_trackerEndcapElements, m_detectorScene );
   ids = geom->getMatchedIds("tid:TID");
   for ( std::vector<unsigned int>::const_iterator id = ids.begin();
         id != ids.end(); ++id ) {
      TEveGeoShape* shape = geom->getShape( *id );
      if ( !shape ) continue;
      shape->SetMainTransparency(m_geomTransparency.value());
      m_trackerEndcapElements->AddElement( shape );
   }
   ids = geom->getMatchedIds("tec:TEC");
   for ( std::vector<unsigned int>::const_iterator id = ids.begin();
         id != ids.end(); ++id ) {
      TEveGeoShape* shape = geom->getShape( *id );
      if ( !shape ) continue;
      shape->SetMainTransparency(m_geomTransparency.value());
      m_trackerEndcapElements->AddElement( shape );
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

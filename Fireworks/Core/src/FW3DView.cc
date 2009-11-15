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
// $Id: FW3DView.cc,v 1.14 2009/04/27 16:53:29 dmytro Exp $
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
#include "DataFormats/Math/interface/deltaR.h"

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
   m_geometry(0),
   m_muonBarrelElements(0),
   m_muonEndcapElements(0),
   m_pixelBarrelElements(0),
   m_pixelEndcapElements(0),
   m_trackerBarrelElements(0),
   m_trackerEndcapElements(0),
   m_showMuonBarrel(this, "Show Muon Barrel", false ),
   m_showMuonEndcap(this, "Show Muon Endcap", false),
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
   m_detectorScene->Destroy();
   m_viewer->DestroyWindowAndSlot();
}

void FW3DView::setGeometry(const DetIdToMatrix* geom )
{
   assert(geom);
   m_geometry = (DetIdToMatrix*) geom;
}

//______________________________________________________________________________
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

//______________________________________________________________________________
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


//==============================================================================
// GEOMETRY
//==============================================================================
void
FW3DView::showMuonBarrel( )
{
   if (!m_muonBarrelElements && m_showMuonBarrel.value())
   {
      m_muonBarrelElements = new TEveElementList( "DT" );
      m_detectorScene->AddElement(m_muonBarrelElements);
      for ( Int_t iWheel = -2; iWheel <= 2; ++iWheel)
      {
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
               TEveGeoShape* shape = m_geometry->getShape( id.rawId() );
               if ( !shape ) continue;
               shape->SetMainTransparency(m_geomTransparency.value());
               cStation->AddElement(shape);
            }
         }
      }
   }

   if (m_muonBarrelElements)
   {
      m_muonBarrelElements->SetRnrState(m_showMuonBarrel.value());
      m_viewer->GetGLViewer()->UpdateScene();
   }
}

//______________________________________________________________________________
void
FW3DView::showMuonEndcap( )
{
   if ( m_showMuonEndcap.value() && !m_muonEndcapElements )
   {
      m_muonEndcapElements = new TEveElementList( "CSC" );
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
                     TEveGeoShape* shape = m_geometry->getShape( id.rawId() );
                     if ( !shape ) continue;
                     shape->SetMainTransparency(m_geomTransparency.value());
                     cRing->AddElement( shape );
                  }
                  catch (... ) {}
               }
            }
         }
      }
   }

   if (m_muonEndcapElements)
   {
      m_muonEndcapElements->SetRnrState(m_showMuonEndcap.value());
      m_viewer->GetGLViewer()->UpdateScene();
   }
}

//______________________________________________________________________________
void
FW3DView::showPixelBarrel( )
{
   if ( m_showPixelBarrel.value() && !m_pixelBarrelElements )
   {
      m_pixelBarrelElements = new TEveElementList( "PixelBarrel" );
      m_pixelBarrelElements->SetRnrState(m_showPixelBarrel.value());
      gEve->AddElement( m_pixelBarrelElements, m_detectorScene );
      std::vector<unsigned int> ids = m_geometry->getMatchedIds("PixelBarrel");
      for ( std::vector<unsigned int>::const_iterator id = ids.begin();
            id != ids.end(); ++id ) {
         TEveGeoShape* shape = m_geometry->getShape( *id );
         if ( !shape ) continue;
         shape->SetMainTransparency(m_geomTransparency.value());
         m_pixelBarrelElements->AddElement( shape );
      }
   }

   if (m_pixelBarrelElements)
   {
      m_pixelBarrelElements->SetRnrState(m_showPixelBarrel.value());
      m_viewer->GetGLViewer()->UpdateScene();
   }
}

//______________________________________________________________________________
void
FW3DView::showPixelEndcap( )
{
   if ( m_showPixelEndcap.value() && ! m_pixelEndcapElements )
   {
      m_pixelEndcapElements = new TEveElementList( "PixelEndcap" );
      gEve->AddElement( m_pixelEndcapElements, m_detectorScene );
      std::vector<unsigned int> ids = m_geometry->getMatchedIds("PixelForward");
      for ( std::vector<unsigned int>::const_iterator id = ids.begin();
            id != ids.end(); ++id ) {
         TEveGeoShape* shape = m_geometry->getShape( *id );
         if ( !shape ) continue;
         shape->SetMainTransparency(m_geomTransparency.value());
         m_pixelEndcapElements->AddElement( shape );
      }
   }

   if (m_pixelEndcapElements)
   {
      m_pixelEndcapElements->SetRnrState(m_showPixelEndcap.value());
      m_viewer->GetGLViewer()->UpdateScene();
   }
}

//______________________________________________________________________________
void
FW3DView::showTrackerBarrel( )
{
   if (  m_showTrackerBarrel.value() &&  !m_trackerBarrelElements )
   {
      m_trackerBarrelElements = new TEveElementList( "TrackerBarrel" );
      m_trackerBarrelElements->SetRnrState(m_showTrackerBarrel.value());
      gEve->AddElement( m_trackerBarrelElements, m_detectorScene );
      std::vector<unsigned int> ids = m_geometry->getMatchedIds("tib:TIB");
      for ( std::vector<unsigned int>::const_iterator id = ids.begin();
            id != ids.end(); ++id ) {
         TEveGeoShape* shape = m_geometry->getShape( *id );
         if ( !shape ) continue;
         shape->SetMainTransparency(m_geomTransparency.value());
         m_trackerBarrelElements->AddElement( shape );
      }
      ids = m_geometry->getMatchedIds("tob:TOB");
      for ( std::vector<unsigned int>::const_iterator id = ids.begin();
            id != ids.end(); ++id ) {
         TEveGeoShape* shape = m_geometry->getShape( *id );
         if ( !shape ) continue;
         shape->SetMainTransparency(m_geomTransparency.value());
         m_trackerBarrelElements->AddElement( shape );
      }
   }

   if (m_trackerBarrelElements )
   {
      m_trackerBarrelElements->SetRnrState(m_showTrackerBarrel.value());
      m_viewer->GetGLViewer()->UpdateScene();
   }
}

//______________________________________________________________________________
void
FW3DView::showTrackerEndcap( )
{
   if ( m_showTrackerEndcap.value() && !m_trackerEndcapElements )
   {
      m_trackerEndcapElements = new TEveElementList( "TrackerEndcap" );
      gEve->AddElement( m_trackerEndcapElements, m_detectorScene );
      std::vector<unsigned int> ids = m_geometry->getMatchedIds("tid:TID");
      for ( std::vector<unsigned int>::const_iterator id = ids.begin();
            id != ids.end(); ++id ) {
         TEveGeoShape* shape = m_geometry->getShape( *id );
         if ( !shape ) continue;
         shape->SetMainTransparency(m_geomTransparency.value());
         m_trackerEndcapElements->AddElement( shape );
      }
      ids = m_geometry->getMatchedIds("tec:TEC");
      for ( std::vector<unsigned int>::const_iterator id = ids.begin();
            id != ids.end(); ++id ) {
         TEveGeoShape* shape = m_geometry->getShape( *id );
         if ( !shape ) continue;
         shape->SetMainTransparency(m_geomTransparency.value());
         m_trackerEndcapElements->AddElement( shape );
      }
   }

   if (m_trackerEndcapElements )
   {
      m_trackerEndcapElements->SetRnrState(m_showTrackerEndcap.value());
      m_viewer->GetGLViewer()->UpdateScene();
   }
}
//==============================================================================
//==============================================================================
//==============================================================================
//==============================================================================

void
FW3DView::showWireFrame( )
{
   if ( m_showWireFrame.value() )
      m_detectorScene->GetGLScene()->SetStyle(TGLRnrCtx::kWireFrame);
   else
      m_detectorScene->GetGLScene()->SetStyle(TGLRnrCtx::kFill);
   m_embeddedViewer->RequestDraw(TGLRnrCtx::kLODHigh);
}

void
FW3DView::setTransparency( )
{
   if ( m_muonBarrelElements ) {
      TEveElementIter iter(m_muonBarrelElements);
      while ( TEveElement* element = iter.current() ) {
         element->SetMainTransparency(m_geomTransparency.value());
         iter.next();
      }
   }
   if ( m_muonEndcapElements ) {
      TEveElementIter iter(m_muonEndcapElements);
      while ( TEveElement* element = iter.current() ) {
         element->SetMainTransparency(m_geomTransparency.value());
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

//______________________________________________________________________________

void
FW3DView::setBackgroundColor(Color_t iColor) {
   m_viewer->GetGLViewer()->SetClearColor(iColor);
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

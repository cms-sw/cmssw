// -*- C++ -*-
//
// Package:     Core
// Class  :     FW3DViewBase
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 21 11:22:41 EST 2008
// $Id: FW3DViewBase.cc,v 1.1 2010/04/07 16:56:20 amraktad Exp $
//
#include <boost/bind.hpp>

// user include files

#include "TGLScenePad.h"
#include "TGLViewer.h"
#include "TGLPerspectiveCamera.h"

#include "TEveManager.h"
#include "TEveElement.h"
#include "TEveScene.h"
#include "TEveViewer.h"
#include "TEveGeoNode.h"

#include "Fireworks/Core/interface/FW3DViewBase.h"
#include "Fireworks/Core/interface/FWEveValueScaler.h"
#include "Fireworks/Core/interface/FWConfiguration.h"
#include "Fireworks/Core/interface/FW3DViewGeometry.h"
#include "Fireworks/Core/interface/TEveElementIter.h"
#include "Fireworks/Core/interface/Context.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//
//double FW3DViewBase::m_scale = 1;

//
// constructors and destructor
//
FW3DViewBase::FW3DViewBase(TEveWindowSlot* iParent, TEveScene* eventScene) :
   FWEveView(iParent),
   m_geometry(0),
   m_showMuonBarrel(this, "Show Muon Barrel", false ),
   m_showMuonEndcap(this, "Show Muon Endcap", false),
   m_showPixelBarrel(this, "Show Pixel Barrel", false ),
   m_showPixelEndcap(this, "Show Pixel Endcap", false),
   m_showTrackerBarrel(this, "Show Tracker Barrel", false ),
   m_showTrackerEndcap(this, "Show Tracker Endcap", false),
   m_showWireFrame(this, "Show Wire Frame", true),
   m_geomTransparency(this,"Detector Transparency", 95l, 0l, 100l),

   m_caloFixedScale(this,"Calo scale (GeV/meter)",2.,0.001,100.),
   m_caloAutoScale (this,"Calo auto scale",true)
{
   setEventScene(eventScene);
   viewer()->AddScene(eventScene);

   //camera
   viewerGL()->SetCurrentCamera(TGLViewer::kCameraPerspXOZ);
}

FW3DViewBase::~FW3DViewBase()
{
}

void FW3DViewBase::setGeometry(fireworks::Context& context)
{
   m_geometry = new FW3DViewGeometry(context.getGeom());
   geoScene()->AddElement(m_geometry);
   
   m_showMuonBarrel.changed_.connect(boost::bind(&FW3DViewGeometry::showMuonBarrel,m_geometry,_1));
   m_showMuonEndcap.changed_.connect(boost::bind(&FW3DViewGeometry::showMuonEndcap,m_geometry,_1));
   m_showPixelBarrel.changed_.connect(boost::bind(&FW3DViewGeometry::showPixelBarrel,m_geometry,_1));
   m_showPixelEndcap.changed_.connect(boost::bind(&FW3DViewGeometry::showPixelEndcap,m_geometry,_1));
   m_showTrackerBarrel.changed_.connect(boost::bind(&FW3DViewGeometry::showTrackerBarrel,m_geometry,_1));
   m_showTrackerEndcap.changed_.connect(boost::bind(&FW3DViewGeometry::showTrackerEndcap,m_geometry,_1));
   m_geomTransparency.changed_.connect(boost::bind(&FW3DViewGeometry::setTransparency,m_geometry, _1));
   m_showWireFrame.changed_.connect(boost::bind(&FW3DViewBase::showWireFrame,this, _1));

}

void
FW3DViewBase::showWireFrame( bool x)
{
   geoScene()->GetGLScene()->SetStyle(x ? TGLRnrCtx::kWireFrame : TGLRnrCtx::kFill);  
   viewerGL()->RequestDraw();
}

//______________________________________________________________________________
void
FW3DViewBase::addTo(FWConfiguration& iTo) const
{
   // take care of parameters
   FWEveView::addTo(iTo);
   TGLPerspectiveCamera* camera = dynamic_cast<TGLPerspectiveCamera*>(&(viewerGL()->CurrentCamera()));
   addToPerspectiveCamera(camera, "Plain3D", iTo);   
}

//______________________________________________________________________________
void
FW3DViewBase::setFrom(const FWConfiguration& iFrom)
{
   // take care of parameters
   FWEveView::setFrom(iFrom);
   TGLPerspectiveCamera* camera = dynamic_cast<TGLPerspectiveCamera*>(&(viewerGL()->CurrentCamera()));
   setFromPerspectiveCamera(camera, "Plain3D", iFrom);

}






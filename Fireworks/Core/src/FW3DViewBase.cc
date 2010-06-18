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
// $Id: FW3DViewBase.cc,v 1.5 2010/05/03 15:47:38 amraktad Exp $
//
#include <boost/bind.hpp>

// user include files

#include "TGLScenePad.h"
#include "TGLViewer.h"
#include "TGLPerspectiveCamera.h"
#include "TEveElement.h"
#include "TEveScene.h"

#include "Fireworks/Core/interface/FW3DViewBase.h"
#include "Fireworks/Core/interface/FW3DViewGeometry.h"
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
FW3DViewBase::FW3DViewBase(TEveWindowSlot* iParent, FWViewType::EType typeId):
   FWEveView(iParent, typeId),
   m_geometry(0),
   m_showMuonBarrel(this, "Show Muon Barrel", false ),
   m_showMuonEndcap(this, "Show Muon Endcap", false),
   m_showPixelBarrel(this, "Show Pixel Barrel", false ),
   m_showPixelEndcap(this, "Show Pixel Endcap", false),
   m_showTrackerBarrel(this, "Show Tracker Barrel", false ),
   m_showTrackerEndcap(this, "Show Tracker Endcap", false),
   m_showWireFrame(this, "Show Wire Frame", true),
   m_geomTransparency(this,"Detector Transparency", 95l, 0l, 100l){
   viewerGL()->SetCurrentCamera(TGLViewer::kCameraPerspXOZ);
}

FW3DViewBase::~FW3DViewBase()
{
}

void FW3DViewBase::setContext(fireworks::Context& context)
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






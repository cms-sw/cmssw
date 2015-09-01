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
//
#include <boost/bind.hpp>

// user include files

#include "TGButton.h"
#include "TGLScenePad.h"
#include "TGLViewer.h"
#include "TGLClip.h"
#include "TGLPerspectiveCamera.h"
#include "TEveManager.h"
#include "TEveElement.h"
#include "TEveLine.h"
#include "TEveBoxSet.h"
#include "TEveScene.h"
#include "TGLLogicalShape.h"
#include "TEveCalo.h"
#include "TEveCaloData.h"
#include "TEveStraightLineSet.h"

#include "Fireworks/Core/interface/FW3DViewBase.h"
#include "Fireworks/Core/interface/FW3DViewGeometry.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWViewContext.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"
#include "Fireworks/Core/interface/CmsShowViewPopup.h"
#include "Fireworks/Core/src/FW3DViewDistanceMeasureTool.h"
#include "Fireworks/Core/interface/FWGeometry.h"

namespace {
class TGLClipsiLogical : public TGLLogicalShape
{
protected:
   virtual void DirectDraw(TGLRnrCtx & rnrCtx) const override{}

public:
   TGLClipsiLogical() : TGLLogicalShape() {}
   virtual ~TGLClipsiLogical() {}
   void Resize(Double_t ext){}
};

const float fgColor[4] = { 1.0, 0.6, 0.2, 0.5 };

class Clipsi : public TGLClip
{
private:
   TGLRnrCtx* m_rnrCtx;
   Clipsi(const Clipsi&);            // Not implemented
   Clipsi& operator=(const Clipsi&); // Not implemented

   TGLVertex3 vtx[4];

public:
   Clipsi(TGLRnrCtx* ctx):TGLClip(* new TGLClipsiLogical, TGLMatrix(), fgColor), m_rnrCtx(ctx){}
   virtual ~Clipsi() {}
   using TGLClip::Setup;
   virtual void Setup(const TGLBoundingBox & bbox) override {}

   void SetPlaneInfo(TEveVector* vec)
   {
      for (int i = 0; i < 4; ++i) {
         // vec[i].Dump();
         vtx[i].Set(vec[i].fX, vec[i].fY, vec[i].fZ);
      }
   }

   using TGLClip::PlaneSet;
   virtual void PlaneSet(TGLPlaneSet_t & planeSet) const override
   {
      TGLVertex3 o;
      planeSet.push_back(TGLPlane(o, vtx[0], vtx[1]));
      planeSet.push_back(TGLPlane(o, vtx[1], vtx[2]));
      planeSet.push_back(TGLPlane(o, vtx[2], vtx[3]));
      planeSet.push_back(TGLPlane(o, vtx[3], vtx[0])); 
   }
};
}

////////////////////////////////////////////////////////////////////////////////
// 
//
//                  FW3DViewBase
//
// 
//
////////////////////////////////////////////////////////////////////////////////
FW3DViewBase::FW3DViewBase(TEveWindowSlot* iParent, FWViewType::EType typeId, unsigned int version):
   FWEveView(iParent, typeId, version ),
   m_geometry(0),
   m_glClip(0),
   m_showMuonBarrel(this, "Show Muon Barrel",  0l, 0l, 2l ),
   m_showMuonEndcap(this, "Show Muon Endcap", false ),
   m_showPixelBarrel(this, "Show Pixel Barrel", false ),
   m_showPixelEndcap(this, "Show Pixel Endcap", false),
   m_showTrackerBarrel(this, "Show Tracker Barrel", false ),
   m_showTrackerEndcap(this, "Show Tracker Endcap", false),
   m_ecalBarrel(0),
   m_showEcalBarrel(this, "Show Ecal Barrel", typeId == FWViewType::kISpy ? true : false),
   m_rnrStyle(this, "Render Style", 0l, 0l, 2l),
   m_selectable(this, "Enable Tooltips", false),
   m_cameraType(this, "Camera Type", 0l, 0l, 5l),
   m_clipEnable(this, "Enable Clip", false),
   m_clipTheta(this, "Clip Theta", 0.0, -5.0, 5.0),
   m_clipPhi(this, "Clip Phi", 0.0, -2.0, 2.0),
   m_clipDelta1(this, "Clip Delta1", 0.2, 0.01, 2),
   m_clipDelta2(this, "Clip Delta2", 0.2, 0.01, 2),
   m_DMT(0),
   m_DMTline(0)
{
   viewerGL()->SetCurrentCamera(TGLViewer::kCameraPerspXOZ);
   m_DMT = new FW3DViewDistanceMeasureTool();

   m_showMuonBarrel.addEntry(0, "Hide");
   m_showMuonBarrel.addEntry(1, "Simplified");
   m_showMuonBarrel.addEntry(2, "Full");
   m_showMuonBarrel.changed_.connect(boost::bind(&FW3DViewBase::showMuonBarrel,this,_1));

   m_rnrStyle.addEntry(TGLRnrCtx::kFill, "Fill");
   m_rnrStyle.addEntry(TGLRnrCtx::kOutline, "Outline");
   m_rnrStyle.addEntry(TGLRnrCtx::kWireFrame, "WireFrame");
   m_rnrStyle.changed_.connect(boost::bind(&FW3DViewBase::rnrStyle,this, _1));

   m_selectable.changed_.connect(boost::bind(&FW3DViewBase::selectable,this, _1));


   m_cameraType.addEntry(TGLViewer::kCameraPerspXOZ,"PerspXOZ" );
   m_cameraType.addEntry(TGLViewer::kCameraOrthoXOY,"OrthoXOY");
   m_cameraType.addEntry(TGLViewer::kCameraOrthoXOZ,"OrthoXOZ");
   m_cameraType.addEntry(TGLViewer::kCameraOrthoZOY,"OrthoZOY" );  
   m_cameraType.addEntry(TGLViewer::kCameraOrthoXnOY,"OrthoXnOY");
   m_cameraType.addEntry(TGLViewer::kCameraOrthoXnOZ,"OrthoXnOZ");
   m_cameraType.addEntry(TGLViewer::kCameraOrthoZnOY,"OrthoZnOY" );  
   m_cameraType.changed_.connect(boost::bind(&FW3DViewBase::setCameraType,this, _1));

   m_clipEnable.changed_.connect(boost::bind(&FW3DViewBase::enableSceneClip,this, _1));
   m_clipTheta.changed_.connect(boost::bind(&FW3DViewBase::updateClipPlanes,this, false));
   m_clipPhi.changed_.connect(boost::bind(&FW3DViewBase::updateClipPlanes,this, false));
   m_clipDelta1.changed_.connect(boost::bind(&FW3DViewBase::updateClipPlanes,this, false));
   m_clipDelta2.changed_.connect(boost::bind(&FW3DViewBase::updateClipPlanes,this, false));


    m_ecalBarrel = new TEveBoxSet("ecalBarrel"); 
    m_ecalBarrel->UseSingleColor();
    m_ecalBarrel->SetMainColor(kAzure+10);
    m_ecalBarrel->SetMainTransparency(98);
    geoScene()->AddElement(m_ecalBarrel);
}

FW3DViewBase::~FW3DViewBase()
{
  delete m_glClip;
}

void FW3DViewBase::setContext(const fireworks::Context& context)
{
   FWEveView::setContext(context);

   m_geometry = new FW3DViewGeometry(context);
   geoScene()->AddElement(m_geometry);
   
   m_showPixelBarrel.changed_.connect(boost::bind(&FW3DViewGeometry::showPixelBarrel,m_geometry,_1));
   m_showPixelEndcap.changed_.connect(boost::bind(&FW3DViewGeometry::showPixelEndcap,m_geometry,_1));
   m_showTrackerBarrel.changed_.connect(boost::bind(&FW3DViewGeometry::showTrackerBarrel,m_geometry,_1));
   m_showTrackerEndcap.changed_.connect(boost::bind(&FW3DViewGeometry::showTrackerEndcap,m_geometry,_1));
   m_showMuonEndcap.changed_.connect(boost::bind(&FW3DViewGeometry::showMuonEndcap,m_geometry,_1));
   m_showEcalBarrel.changed_.connect(boost::bind(&FW3DViewBase::showEcalBarrel, this,_1));

   // don't clip event scene --  ideally, would have TGLClipNoClip in root
   TGLClipPlane* c=new TGLClipPlane();
   c->Setup(TGLVector3(1e10,0,0), TGLVector3(-1,0,0));
   eventScene()->GetGLScene()->SetClip(c);

   m_DMTline = new TEveLine();
   m_DMTline->SetLineColor(1016);
   m_DMTline->SetLineStyle(5);


   m_DMTline->SetPoint(0, 0, 0, 0);
   m_DMTline->SetPoint(1, 0, 0, 0);
   eventScene()->AddElement(m_DMTline);
   showEcalBarrel(m_showEcalBarrel.value());
}

void FW3DViewBase::showMuonBarrel(long x)
{
   if (m_geometry)
   {
      m_geometry->showMuonBarrel(x == 1);
      m_geometry->showMuonBarrelFull(x == 2);
   }
}

void FW3DViewBase::setCameraType(long x)
{
   viewerGL()->RefCamera(TGLViewer::ECameraType(x)).IncTimeStamp();
   viewerGL()->SetCurrentCamera(TGLViewer::ECameraType(x));
   
   //if (viewerGL()->CurrentCamera().IsOrthographic())
   //   ((TGLOrthoCamera*)(&viewerGL()->CurrentCamera()))->SetEnableRotate(1);
}

void
FW3DViewBase::rnrStyle( long x)
{
   geoScene()->GetGLScene()->SetStyle(x);  
   viewerGL()->Changed();
   gEve->Redraw3D();
}

void
FW3DViewBase::selectable( bool x)
{
      geoScene()->GetGLScene()->SetSelectable(x);
}
void
FW3DViewBase::enableSceneClip( bool x)
{
   if (m_glClip == 0)  {
      m_glClip = new Clipsi(viewerGL()->GetRnrCtx());

      m_glClip->SetMode(TGLClip::kOutside);
   }

   geoScene()->GetGLScene()->SetClip(x ? m_glClip : 0);
   for (TEveElement::List_i it =gEve->GetScenes()->BeginChildren(); it != gEve->GetScenes()->EndChildren(); ++it )
   {
      if (strncmp((*it)->GetElementName(), "TopGeoNodeScene", 15) == 0)
         ((TEveScene*)(*it))->GetGLScene()->SetClip(x ? m_glClip : 0);
   }
   eventScene()->GetGLScene()->SetClip(x ? m_glClip : 0);
   updateClipPlanes(true);
   viewerGL()->RequestDraw();
}

void
FW3DViewBase::setClip(float theta, float phi)
{
   // called from popup menu via FWGUIManager
   
   // limit to 2 decimals, else TGNumber entry in the view controller shows only last 5 irrelevant digits
   double base = 100.0;
   int thetaInt = theta*base;
   int phiInt = phi*base;
   m_clipTheta.set(thetaInt/base);
   m_clipPhi.set(phiInt/base);
   m_clipEnable.set(true);
}

namespace {
float getBBoxLineLength(TEveScene* scene, TEveVector in)
{
    if (!scene->NumChildren()) return 0;

    scene->Repaint();
    scene->GetGLScene()->CalcBoundingBox();
    const TGLBoundingBox& bb = scene->GetGLScene()->BoundingBox();
    if (bb.IsEmpty()) return 0;

    TGLPlaneSet_t ps; bb.PlaneSet(ps);
    TEveVector inn = in; inn.Normalize();
    inn *= 10000;
    TGLLine3 line(TGLVertex3(), TGLVertex3(inn.fX, inn.fY, inn.fZ));
    std::vector<float> res;
    for (TGLPlaneSet_i i = ps.begin(); i!= ps.end(); ++i)
    {
        std::pair<Bool_t, TGLVertex3> r = Intersection(*i, line, false);
        if(r.first) {
            TGLVector3 vr(r.second.X(), r.second.Y(), r.second.Z());
            res.push_back(vr.Mag());
        }
    }
    std::sort(res.begin(), res.end());
    return res.front();
}

void setBBoxClipped(TGLBoundingBox& bbox, TEveVector dir, TEveVector b0, TEveVector b1, float fac)
{
    dir *= fac;
    b0 *= fac;
    b1 *= fac;

    TEveVectorD bb[8];
    bb[0] += b0; bb[0] += b1; 
    bb[1] -= b0; bb[1] += b1;  
    bb[2] -= b0; bb[2] -= b1;  
    bb[3] += b0; bb[3] -= b1; 

    for (int i = 4; i < 8; ++i)
        bb[i] = dir;

    bb[0+4] += b0; bb[0+4] += b1; 
    bb[1+4] -= b0; bb[1+4] += b1;  
    bb[2+4] -= b0; bb[2+4] -= b1;  
    bb[3+4] += b0; bb[3+4] -= b1; 

    TGLVertex3 bbv[8];
    for (int i = 0; i < 8; ++i) {
        bbv[i].Set(bb[i].fX, bb[i].fY, bb[i].fZ);
    }
    bbox.Set(bbv);
}
}

void
FW3DViewBase::updateClipPlanes(bool resetCamera)
{
   //  TEveScene* gs = (TEveScene*)gEve->GetScenes()->FindChild(TString("TopGeoNodeScene"));
   //printf("node scene %p\n", gs);
   if (m_clipEnable.value())
   {
      float theta = m_clipTheta.value();
      float phi   = m_clipPhi.value();
      using namespace TMath;
      TEveVector in(Sin(theta)*Cos(phi), Sin(theta)*Sin(phi), Cos(theta));

      // one side of cross section plane is paralel to XY plane
      TEveVector normXY(0., 1., 0);
      TEveVector b0 = in.Cross(normXY);
      TEveVector b1 = in.Cross(b0);

      float delta1   = m_clipDelta1.value();
      float delta2   = m_clipDelta2.value();
      b0.Normalize();
      b0 *= Sin(delta1);
      b1.Normalize();
      b1 *= Sin(delta2);
   
      TEveVector c[4];
      c[0] += b0; c[0] += b1; 
      c[1] -= b0; c[1] += b1;  
      c[2] -= b0; c[2] -= b1;  
      c[3] += b0; c[3] -= b1; 
      for (int i = 0; i < 4; ++i)
          c[i] += in;

      ((Clipsi*)m_glClip)->SetPlaneInfo(&c[0]);


      if (resetCamera) {
      TGLBoundingBox bbox;
      float es = getBBoxLineLength(eventScene(), in);
      float gs = getBBoxLineLength(geoScene(), in);
      setBBoxClipped(bbox, in, b0, b1, TMath::Max(es, gs));

      /*
    TEvePointSet* bmarker = new TEvePointSet(8);
    bmarker->Reset(4);
    bmarker->SetName("bbox");
    bmarker->SetMarkerColor(kOrange);
    bmarker->SetMarkerStyle(3);
    bmarker->SetMarkerSize(0.2);
    for (int i = 0; i < 8; ++i)
        bmarker->SetPoint(i, bbox[i].X(), bbox[i].Y(), bbox[i].Z());
    eventScene()->AddElement(bmarker); 
      */


      TGLCamera& cam = viewerGL()->CurrentCamera();
      cam.SetExternalCenter(true);
      cam.SetCenterVec(bbox.Center().X(), bbox.Center().Y(), bbox.Center().Z());
      cam.Setup(bbox, true);
      }
      else {
         eventScene()->Repaint();
      }
         }
   
   gEve->Redraw3D();

}

//______________________________________________________________________________
void
FW3DViewBase::addTo(FWConfiguration& iTo) const
{
   // take care of parameters
   FWEveView::addTo(iTo);
   TGLPerspectiveCamera* camera = dynamic_cast<TGLPerspectiveCamera*>(&(viewerGL()->CurrentCamera()));
   if (camera)
      addToPerspectiveCamera(camera, "Plain3D", iTo);   
}

//______________________________________________________________________________
void
FW3DViewBase::setFrom(const FWConfiguration& iFrom)
{
   // take care of parameters
   FWEveView::setFrom(iFrom);

   TGLPerspectiveCamera* camera = dynamic_cast<TGLPerspectiveCamera*>(&(viewerGL()->CurrentCamera()));
   if (camera)
      setFromPerspectiveCamera(camera, "Plain3D", iFrom);

   if (iFrom.version() < 5)
   {
      // transparency moved to common preferences in FWEveView version 5
      std::string tName("Detector Transparency");
      std::istringstream s(iFrom.valueForKey(tName)->value());
      int transp;
      s>> transp;
      context().colorManager()->setGeomTransparency(transp, false);
   }
}

bool FW3DViewBase::requestGLHandlerPick() const
{
   return m_DMT->m_action != FW3DViewDistanceMeasureTool::kNone;
}

void FW3DViewBase::setCurrentDMTVertex(double x, double y, double z)
{
  if (m_DMT->m_action == FW3DViewDistanceMeasureTool::kNone)
    printf( "ERROR!!!! FW3DViewBase::setCurrentDMTVertex \n");

   m_DMTline->SetPoint(m_DMT->m_action, x, y, z);
   m_DMTline->ElementChanged();
   viewerGL()->RequestDraw();

   m_DMT->refCurrentVertex().Set(x, y,z);
   m_DMT->resetAction();
}

void 
FW3DViewBase::populateController(ViewerParameterGUI& gui) const
{
   FWEveView::populateController(gui);

   gui.requestTab("Detector").
      addParam(&m_showMuonBarrel).
      addParam(&m_showMuonEndcap).
      addParam(&m_showTrackerBarrel).
      addParam(&m_showTrackerEndcap).
      addParam(&m_showPixelBarrel).
      addParam(&m_showPixelEndcap).  
      addParam(&m_showEcalBarrel).  
      addParam(&m_rnrStyle).
      addParam(&m_selectable).
      separator().
      addParam(&m_clipEnable).
      addParam(&m_clipTheta).
      addParam(&m_clipPhi).
      addParam(&m_clipDelta1).
      addParam(&m_clipDelta2);


   gui.requestTab("Style").separator();
   gui.getTabContainer()->AddFrame(new TGTextButton(gui.getTabContainer(), "Root controls",
                                                    Form("TEveGedEditor::SpawnNewEditor((TGLViewer*)0x%lx)", (unsigned long)viewerGL())));

   gui.requestTab("Tools").addParam(&m_cameraType).separator();
   gui.getTabContainer()->AddFrame(m_DMT->buildGUI( gui.getTabContainer()), new TGLayoutHints(kLHintsExpandX, 2, 2, 2, 2));
}

void  FW3DViewBase::showEcalBarrel(bool x) {
    if (x &&  m_ecalBarrel->GetPlex()->Size() == 0) {
        const FWGeometry* geom = context().getGeom();
        std::vector<unsigned int> ids = geom->getMatchedIds(FWGeometry::Detector::Ecal, FWGeometry::SubDetector::PixelBarrel);
        m_ecalBarrel->Reset(TEveBoxSet::kBT_FreeBox, true, ids.size() );
        for (std::vector<unsigned int>::iterator it = ids.begin(); it != ids.end(); ++it) {
            const float* cor = context().getGeom()->getCorners(*it);
            m_ecalBarrel->AddBox(cor);
        }
        m_ecalBarrel->RefitPlex();
    }

    if (m_ecalBarrel->GetRnrSelf() != x) {
        m_ecalBarrel->SetRnrSelf(x);
        gEve->Redraw3D();
    }

    // disable enable grid in 3DView
    if (typeId() == FWViewType::k3D) {
       TEveElement* calo = eventScene()->FindChild("calo barrel");
       if (calo) {
          TEveCalo3D* c3d = dynamic_cast<TEveCalo3D*>(calo);
          c3d->SetRnrFrame(!x, !x);
          c3d->ElementChanged();
       }
    }
}

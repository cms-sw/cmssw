// -*- C++ -*-
//
// Package:     Vertexs
// Class  :     FWVertexProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Dec  2 14:17:03 EST 2008
//
// user include files// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWProxyBuilderConfiguration.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/CmsShowCommon.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWParameters.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "Fireworks/Vertices/interface/TEveEllipsoid.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "TEvePointSet.h"
#include "TMatrixDEigen.h"
#include "TMatrixDSym.h"
#include "TDecompSVD.h"
#include "TVectorD.h"
#include "TEveTrans.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "TEveStraightLineSet.h"
#include "TEveBoxSet.h"
#include "TGeoSphere.h"
#include "TEveGeoNode.h"
#include "TEveGeoShape.h"
#include "TEveVSDStructs.h"

class FWVertexProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Vertex> {
public:
  FWVertexProxyBuilder() {}
  ~FWVertexProxyBuilder() override {}

  void setItem(const FWEventItem* iItem) override {
    FWProxyBuilderBase::setItem(iItem);
    if (iItem) {
      iItem->getConfig()->assertParam("Draw Tracks", false);
      iItem->getConfig()->assertParam("Draw Pseudo Track", false);
      iItem->getConfig()->assertParam("Draw Ellipse", false);
      iItem->getConfig()->assertParam("Scale Ellipse", 2l, 1l, 10l);
      iItem->getConfig()->assertParam(
          "Ellipse Color Index", 6l, 0l, (long)context().colorManager()->numberOfLimitedColors());
      iItem->getConfig()->assertParam("Event Center Index", -1l, -1l, 500l);
    }
  }

  REGISTER_PROXYBUILDER_METHODS();

  FWVertexProxyBuilder(const FWVertexProxyBuilder&) = delete;                   // stop default
  const FWVertexProxyBuilder& operator=(const FWVertexProxyBuilder&) = delete;  // stop default

private:
  using FWSimpleProxyBuilderTemplate<reco::Vertex>::build;
  void build(const reco::Vertex& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*) override;

  void localModelChanges(const FWModelId& iId,
                         TEveElement* iCompound,
                         FWViewType::EType viewType,
                         const FWViewContext* vc) override;
};

void FWVertexProxyBuilder::build(const reco::Vertex& iData,
                                 unsigned int iIndex,
                                 TEveElement& oItemHolder,
                                 const FWViewContext* vc) {
  const reco::Vertex& v = iData;

  // marker
  TEveGeoManagerHolder gmgr(TEveGeoShape::GetGeoMangeur());
  TEvePointSet* pointSet = new TEvePointSet();
  pointSet->SetNextPoint(v.x(), v.y(), v.z());
  setupAddElement(pointSet, &oItemHolder);

  // ellipse
  if (item()->getConfig()->value<bool>("Draw Ellipse")) {
    TEveEllipsoid* eveEllipsoid = new TEveEllipsoid("Ellipsoid", Form("Ellipsoid %d", iIndex));

    eveEllipsoid->RefPos().Set(v.x(), v.y(), v.z());

    reco::Vertex::Error e = v.error();
    TMatrixDSym m(3);
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) {
        m(i, j) = e(i, j);
        eveEllipsoid->RefEMtx()(i + 1, j + 1) = e(i, j);
      }

    // external scaling
    double ellipseScale = 1.;
    if (item()->getConfig()->value<long>("Scale Ellipse"))
      ellipseScale = item()->getConfig()->value<long>("Scale Ellipse");

    eveEllipsoid->SetScale(ellipseScale);

    // cache 3D extend used in eval bbox and render 3D
    TMatrixDEigen eig(m);
    TVectorD vv(eig.GetEigenValuesRe());
    eveEllipsoid->RefExtent3D().Set(sqrt(vv(0)) * ellipseScale, sqrt(vv(1)) * ellipseScale, sqrt(vv(2)) * ellipseScale);

    eveEllipsoid->SetLineWidth(2);
    setupAddElement(eveEllipsoid, &oItemHolder);
    eveEllipsoid->SetMainTransparency(TMath::Min(100, 80 + item()->defaultDisplayProperties().transparency() / 5));

    Color_t color = item()->getConfig()->value<long>("Ellipse Color Index");
    // eveEllipsoid->SetFillColor(item()->defaultDisplayProperties().color());
    // eveEllipsoid->SetLineColor(item()->defaultDisplayProperties().color());
    eveEllipsoid->SetMainColor(color + context().colorManager()->offsetOfLimitedColors());
  }

  // tracks
  if (item()->getConfig()->value<bool>("Draw Tracks")) {
    for (reco::Vertex::trackRef_iterator it = v.tracks_begin(); it != v.tracks_end(); ++it) {
      float w = v.trackWeight(*it);
      if (w < 0.5)
        continue;

      const reco::Track& track = *it->get();
      TEveRecTrack t;
      t.fBeta = 1.;
      t.fV = TEveVector(track.vx(), track.vy(), track.vz());
      t.fP = TEveVector(track.px(), track.py(), track.pz());
      t.fSign = track.charge();
      TEveTrack* trk = new TEveTrack(&t, context().getTrackPropagator());
      trk->SetMainColor(item()->defaultDisplayProperties().color());
      trk->MakeTrack();
      setupAddElement(trk, &oItemHolder);
    }
  }
  if (item()->getConfig()->value<bool>("Draw Pseudo Track")) {
    TEveRecTrack t;
    t.fBeta = 1.;
    t.fV = TEveVector(v.x(), v.y(), v.z());
    t.fP = TEveVector(-v.p4().px(), -v.p4().py(), -v.p4().pz());
    t.fSign = 1;
    TEveTrack* trk = new TEveTrack(&t, context().getTrackPropagator());
    trk->SetLineStyle(7);
    trk->MakeTrack();
    setupAddElement(trk, &oItemHolder);
  }

  fireworks::Context* context = fireworks::Context::getInstance();
  int cIdx = item()->getConfig()->value<long>("Event Center Index");
  if (cIdx == -1) {
    context->commonPrefs()->resetEventCenter();
  }
  if (cIdx >= 0 && int(iIndex) == int(cIdx)) {
    fwLog(fwlog::kInfo) << "FWVertexProxyBuilder set event center "
                        << Form("idx [%d], (%f %f %f) \n", cIdx, v.x(), v.y(), v.z());
    context->commonPrefs()->setEventCenter(v.x(), v.y(), v.z());
  }
}

void FWVertexProxyBuilder::localModelChanges(const FWModelId& iId,
                                             TEveElement* iCompound,
                                             FWViewType::EType viewType,
                                             const FWViewContext* vc) {
  increaseComponentTransparency(iId.index(), iCompound, "Ellipsoid", 80);
  TEveElement* el = iCompound->FindChild("Ellipsoid");
  if (el)
    el->SetMainColor(item()->getConfig()->value<long>("Ellipse Color Index") +
                     context().colorManager()->offsetOfLimitedColors());
}

//
// static member functions
//
REGISTER_FWPROXYBUILDER(FWVertexProxyBuilder, reco::Vertex, "Vertices", FWViewType::k3DBit | FWViewType::kAllRPZBits);

#include "Fireworks/Core/interface/BuilderUtils.h"
#include "Fireworks/Core/interface/CmsShowCommon.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "Fireworks/Core/interface/FWParameters.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWProxyBuilderConfiguration.h"
#include "Fireworks/Vertices/interface/TEveEllipsoid.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/Math/interface/Vector3D.h"

#include "TEvePointSet.h"
#include "TMatrixDEigen.h"
#include "TMatrixDSym.h"
#include "TDecompSVD.h"
#include "TVectorD.h"
#include "TEveTrans.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "TEveStraightLineSet.h"
#include "TGeoSphere.h"
#include "TEveGeoNode.h"
#include "TEveVSDStructs.h"
#include "TEveBoxSet.h"
#include "TEveGeoShape.h"

class FWTracksterProxyBuilder : public FWSimpleProxyBuilderTemplate<ticl::Trackster> {
public:
  FWTracksterProxyBuilder(void) {}
  ~FWTracksterProxyBuilder(void) override {}

  REGISTER_PROXYBUILDER_METHODS();

private:
  FWTracksterProxyBuilder(const FWTracksterProxyBuilder &) = delete;                   // stop default
  const FWTracksterProxyBuilder &operator=(const FWTracksterProxyBuilder &) = delete;  // stop default

  void build(const ticl::Trackster &iData,
             unsigned int iIndex,
             TEveElement &oItemHolder,
             const FWViewContext *) override;
};

void FWTracksterProxyBuilder::build(const ticl::Trackster &iData,
                                            unsigned int iIndex,
                                            TEveElement &oItemHolder,
                                            const FWViewContext *) {
  const ticl::Trackster &trackster = iData;
  const ticl::Trackster::Vector &barycenter = trackster.barycenter();

  TEveGeoManagerHolder gmgr(TEveGeoShape::GetGeoMangeur());
  TEveEllipsoid* eveEllipsoid = new TEveEllipsoid("Ellipsoid", Form("Ellipsoid %d", iIndex));
  eveEllipsoid->RefPos().Set(barycenter.x(), barycenter.y(), barycenter.z());
  eveEllipsoid->SetScale(1.0);
  eveEllipsoid->SetLineWidth(2);
  setupAddElement(eveEllipsoid, &oItemHolder);
  eveEllipsoid->SetMainTransparency(TMath::Min(100, 80 + item()->defaultDisplayProperties().transparency() / 5));
  Color_t color = item()->getConfig()->value<long>("Ellipse Color Index");
  // eveEllipsoid->SetFillColor(item()->defaultDisplayProperties().color());
  // eveEllipsoid->SetLineColor(item()->defaultDisplayProperties().color());
  eveEllipsoid->SetMainColor(color + context().colorManager()->offsetOfLimitedColors());
}

REGISTER_FWPROXYBUILDER(FWTracksterProxyBuilder,
                        ticl::Trackster,
                        "Trackster",
                        FWViewType::k3DBit | FWViewType::kAllRPZBits);

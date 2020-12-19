#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"

#include "TEveTrans.h"
#include "TGeoSphere.h"
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
  const std::array<float, 3> &eigenvalues = trackster.eigenvalues();
  const double theta = barycenter.Theta();
  const double phi = barycenter.Phi();

  auto eveEllipsoid = new TEveGeoShape("Ellipsoid");
  auto sphere = new TGeoSphere(0., 1.);
  eveEllipsoid->SetShape(sphere);
  eveEllipsoid->InitMainTrans();
  eveEllipsoid->RefMainTrans().Move3PF(barycenter.x(), barycenter.y(), barycenter.z());
  eveEllipsoid->RefMainTrans().SetRotByAnyAngles(theta, phi, 0., "xzy");
  eveEllipsoid->RefMainTrans().SetScale(sqrt(eigenvalues[2]), sqrt(eigenvalues[1]), sqrt(eigenvalues[0]));
  setupAddElement(eveEllipsoid, &oItemHolder);
}

REGISTER_FWPROXYBUILDER(FWTracksterProxyBuilder,
                        ticl::Trackster,
                        "Trackster",
                        FWViewType::k3DBit | FWViewType::kAllRPZBits);

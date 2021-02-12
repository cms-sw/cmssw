#include "Fireworks/Calo/interface/FWHeatmapProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"

#include "TEveBoxSet.h"
#include "TGeoSphere.h"
#include "TGeoTube.h"
#include "TEveGeoShape.h"
#include "TEveStraightLineSet.h"

#include <cmath>

class FWTracksterLayersProxyBuilder : public FWHeatmapProxyBuilderTemplate<ticl::Trackster> {
public:
  FWTracksterLayersProxyBuilder(void) {}
  ~FWTracksterLayersProxyBuilder(void) override {}

  REGISTER_PROXYBUILDER_METHODS();

private:
  edm::Handle<edm::ValueMap<std::pair<float, float>>> TimeValueMapHandle;
  edm::Handle<std::vector<reco::CaloCluster>> layerClustersHandle;
  double timeLowerBound, timeUpperBound;
  long layer;
  double saturation_energy;
  bool heatmap;
  bool z_plus;
  bool z_minus;
  bool enableTimeFilter;
  bool enableSeedLines;
  bool enablePositionLines;
  bool enableEdges;
  double displayMode;
  double proportionalityFactor;

  FWTracksterLayersProxyBuilder(const FWTracksterLayersProxyBuilder &) = delete;                   // stop default
  const FWTracksterLayersProxyBuilder &operator=(const FWTracksterLayersProxyBuilder &) = delete;  // stop default

  void setItem(const FWEventItem *iItem) override;

  void build(const FWEventItem *iItem, TEveElementList *product, const FWViewContext *vc) override;
  void build(const ticl::Trackster &iData,
             unsigned int iIndex,
             TEveElement &oItemHolder,
             const FWViewContext *) override;
};

void FWTracksterLayersProxyBuilder::setItem(const FWEventItem *iItem) {
  FWHeatmapProxyBuilderTemplate::setItem(iItem);
  if (iItem) {
    iItem->getConfig()->assertParam("Cluster(0)/RecHit(1)", false);
    iItem->getConfig()->assertParam("EnableSeedLines", false);
    iItem->getConfig()->assertParam("EnablePositionLines", false);
    iItem->getConfig()->assertParam("EnableEdges", false);
    iItem->getConfig()->assertParam("EnableTimeFilter", false);
    iItem->getConfig()->assertParam("TimeLowerBound(ns)", 0.01, 0.0, 75.0);
    iItem->getConfig()->assertParam("TimeUpperBound(ns)", 0.01, 0.0, 75.0);
    iItem->getConfig()->assertParam("TimeUpperBound(ns)", 0.01, 0.0, 75.0);
    iItem->getConfig()->assertParam("DisplayMode", 0.0, 0.0, 5.0);
    iItem->getConfig()->assertParam("ProportionalityFactor", 1.0, 0.0, 1.0);
  }
}

void FWTracksterLayersProxyBuilder::build(const FWEventItem *iItem, TEveElementList *product, const FWViewContext *vc) {
  iItem->getEvent()->getByLabel(edm::InputTag("hgcalLayerClusters", "timeLayerCluster"), TimeValueMapHandle);
  iItem->getEvent()->getByLabel(edm::InputTag("hgcalLayerClusters"), layerClustersHandle);
  if (TimeValueMapHandle.isValid()) {
    timeLowerBound = std::min(item()->getConfig()->value<double>("TimeLowerBound(ns)"),
                              item()->getConfig()->value<double>("TimeUpperBound(ns)"));
    timeUpperBound = std::max(item()->getConfig()->value<double>("TimeLowerBound(ns)"),
                              item()->getConfig()->value<double>("TimeUpperBound(ns)"));
  } else {
    std::cerr << "Warning: couldn't locate 'timeLayerCluster' ValueMap in root file." << std::endl;
  }

  if (!layerClustersHandle.isValid()) {
    std::cerr << "Warning: couldn't locate 'hgcalLayerClusters' collection in root file." << std::endl;
  }

  layer = item()->getConfig()->value<long>("Layer");
  saturation_energy = item()->getConfig()->value<double>("EnergyCutOff");
  heatmap = item()->getConfig()->value<bool>("Heatmap");
  z_plus = item()->getConfig()->value<bool>("Z+");
  z_minus = item()->getConfig()->value<bool>("Z-");
  enableTimeFilter = item()->getConfig()->value<bool>("EnableTimeFilter");
  enableSeedLines = item()->getConfig()->value<bool>("EnableSeedLines");
  enablePositionLines = item()->getConfig()->value<bool>("EnablePositionLines");
  enableEdges = item()->getConfig()->value<bool>("EnableEdges");
  displayMode = item()->getConfig()->value<double>("DisplayMode");
  proportionalityFactor = item()->getConfig()->value<double>("ProportionalityFactor");

  FWHeatmapProxyBuilderTemplate::build(iItem, product, vc);
}

void FWTracksterLayersProxyBuilder::build(const ticl::Trackster &iData,
                                        unsigned int iIndex,
                                        TEveElement &oItemHolder,
                                        const FWViewContext *) {
  if (enableTimeFilter && TimeValueMapHandle.isValid()) {
    const float time = TimeValueMapHandle->get(iIndex).first;
    if (time < timeLowerBound || time > timeUpperBound)
      return;
  }

  const ticl::Trackster &trackster = iData;
  const size_t N = trackster.vertices().size();
  const std::vector<reco::CaloCluster> &layerClusters = *layerClustersHandle;

  for (size_t i = 0; i < N; ++i) {
    const reco::CaloCluster layerCluster = layerClusters[trackster.vertices(i)];
    const math::XYZPoint &position = layerCluster.position();
    const size_t nHits = layerCluster.size();
    const double energy = layerCluster.correctedEnergy();
    const bool isFine = (HGCSiliconDetId(layerCluster.seed()).type() == 0);
    float radius = 0;
    if (displayMode == 0) {
      radius = sqrt(nHits);
    } else if (displayMode == 1) {
        radius = nHits;
    } else if (displayMode == 2) {
      radius = energy;
    } else if (displayMode == 3) {
      radius = energy/nHits;
    } else if (displayMode == 4) {
      float side = (isFine) ? 0.465 : 0.698;
      float area = pow(side,2)*3*sqrt(3)/2;
      radius = sqrt(nHits*area)/M_PI;
    }

    auto eveCircle = new TEveGeoShape("Circle");
    auto tube = new TGeoTube(0., proportionalityFactor*radius, 0.1);
    eveCircle->SetShape(tube);
    eveCircle->InitMainTrans();
    eveCircle->RefMainTrans().Move3PF(position.x(), position.y(), position.z());
    setupAddElement(eveCircle, &oItemHolder);

    // seed and cluster position
    const float crossScale = 1.0f + fmin(energy, 5.0f);
    if (enablePositionLines) {
      TEveStraightLineSet *position_marker = new TEveStraightLineSet;
      position_marker->SetLineWidth(2);
      position_marker->SetLineColor(kOrange);
      auto const &pos = layerCluster.position();
      const float position_crossScale = crossScale * 0.5;
      position_marker->AddLine(
          pos.x() - position_crossScale, pos.y(), pos.z(), pos.x() + position_crossScale, pos.y(), pos.z());
      position_marker->AddLine(
          pos.x(), pos.y() - position_crossScale, pos.z(), pos.x(), pos.y() + position_crossScale, pos.z());

      oItemHolder.AddElement(position_marker);
    }
  }
}

REGISTER_FWPROXYBUILDER(FWTracksterLayersProxyBuilder, ticl::Trackster, "Trackster layers", FWViewType::kISpyBit);

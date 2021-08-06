#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Fireworks/Calo/interface/FWHeatmapProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
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

  FWTracksterLayersProxyBuilder(const FWTracksterLayersProxyBuilder &) = delete;                   // stop default
  const FWTracksterLayersProxyBuilder &operator=(const FWTracksterLayersProxyBuilder &) = delete;  // stop default

private:
  edm::Handle<edm::ValueMap<std::pair<float, float>>> TimeValueMapHandle;
  edm::Handle<std::vector<reco::CaloCluster>> layerClustersHandle;
  double timeLowerBound, timeUpperBound;
  long layer;
  bool z_plus;
  bool z_minus;
  bool enableTimeFilter;
  bool enablePositionLines;
  bool enableEdges;
  double displayMode;
  double proportionalityFactor;

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
    iItem->getConfig()->assertParam("EnablePositionLines", false);
    iItem->getConfig()->assertParam("EnableEdges", false);
    iItem->getConfig()->assertParam("EnableTimeFilter", false);
    iItem->getConfig()->assertParam("TimeLowerBound(ns)", 0.01, 0.0, 75.0);
    iItem->getConfig()->assertParam("TimeUpperBound(ns)", 0.01, 0.0, 75.0);
    iItem->getConfig()->assertParam("DisplayMode", 0.0, 0.0, 5.0);
    iItem->getConfig()->assertParam("ProportionalityFactor", 1.0, 0.0, 1.0);
  }
}

void FWTracksterLayersProxyBuilder::build(const FWEventItem *iItem, TEveElementList *product, const FWViewContext *vc) {
  iItem->getEvent()->getByLabel(edm::InputTag("hgcalLayerClusters", "timeLayerCluster"), TimeValueMapHandle);
  iItem->getEvent()->getByLabel(edm::InputTag("hgcalLayerClusters"), layerClustersHandle);
  if (TimeValueMapHandle.isValid()) {
    timeLowerBound = item()->getConfig()->value<double>("TimeLowerBound(ns)");
    timeUpperBound = item()->getConfig()->value<double>("TimeUpperBound(ns)");
    if (timeLowerBound > timeUpperBound) {
      edm::LogWarning("InvalidParameters")
          << "lower time bound is larger than upper time bound. Maybe opposite is desired?";
    }
  } else {
    edm::LogWarning("DataNotFound|InvalidData") << "couldn't locate 'timeLayerCluster' ValueMap in root file.";
  }

  if (!layerClustersHandle.isValid()) {
    edm::LogWarning("DataNotFound|InvalidData") << "couldn't locate 'timeLayerCluster' ValueMap in root file.";
  }

  layer = item()->getConfig()->value<long>("Layer");
  z_plus = item()->getConfig()->value<bool>("Z+");
  z_minus = item()->getConfig()->value<bool>("Z-");
  enableTimeFilter = item()->getConfig()->value<bool>("EnableTimeFilter");
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
    float radius = 0;

    // discard everything thats not at the side that we are intersted in
    const bool z = (layerCluster.seed() >> 25) & 0x1;
    if (((z_plus & z_minus) != 1) && (((z_plus | z_minus) == 0) || !(z == z_minus || z == !z_plus)))
      continue;

    if (displayMode == 0) {
      radius = sqrt(nHits);
    } else if (displayMode == 1) {
      radius = nHits;
    } else if (displayMode == 2) {
      radius = energy;
    } else if (displayMode == 3) {
      radius = energy / nHits;
    } else if (displayMode == 4) {
      const bool isScintillator = layerCluster.seed().det() == DetId::HGCalHSc;
      float area = 0;
      if (isScintillator) {
        const bool isFine = (HGCScintillatorDetId(layerCluster.seed()).type() == 0);
        float dphi = (isFine) ? 1.0 * M_PI / 180. : 1.25 * M_PI / 180.;
        int ir = HGCScintillatorDetId(layerCluster.seed()).iradiusAbs();
        float dr = (isFine) ? (0.0484 * static_cast<float>(ir) + 2.1) : (0.075 * static_cast<float>(ir) + 2.0);
        float r = (isFine) ? (0.0239 * static_cast<float>(pow(ir, 2)) + 2.02 * static_cast<float>(ir) + 119.6)
                           : (0.0367 * static_cast<float>(pow(ir, 2)) + 1.7 * static_cast<float>(ir) + 90.7);
        area = r * dr * dphi;
      } else {
        const bool isFine = (HGCSiliconDetId(layerCluster.seed()).type() == 0);
        float side = (isFine) ? 0.465 : 0.698;
        area = pow(side, 2) * 3 * sqrt(3) / 2;
      }
      radius = sqrt(nHits * area) / M_PI;
    }

    auto eveCircle = new TEveGeoShape("Circle");
    auto tube = new TGeoTube(0., proportionalityFactor * radius, 0.1);
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

  if (enableEdges) {
    auto &edges = trackster.edges();

    for (auto edge : edges) {
      auto doublet = std::make_pair(layerClusters[edge[0]], layerClusters[edge[1]]);

      const bool isScintillatorIn = doublet.first.seed().det() == DetId::HGCalHSc;
      const bool isScintillatorOut = doublet.second.seed().det() == DetId::HGCalHSc;
      int layerIn = (isScintillatorIn) ? (HGCScintillatorDetId(doublet.first.seed()).layer())
                                       : (HGCSiliconDetId(doublet.first.seed()).layer());
      int layerOut = (isScintillatorOut) ? (HGCScintillatorDetId(doublet.second.seed()).layer())
                                         : (HGCSiliconDetId(doublet.second.seed()).layer());

      // Check if offset is needed
      const int offset = 28;
      const int offsetIn = offset * (doublet.first.seed().det() != DetId::HGCalEE);
      const int offsetOut = offset * (doublet.second.seed().det() != DetId::HGCalEE);
      layerIn += offsetIn;
      layerOut += offsetOut;

      const bool isAdjacent = (layerOut - layerIn) == 1;

      TEveStraightLineSet *marker = new TEveStraightLineSet;
      marker->SetLineWidth(2);
      if (isAdjacent) {
        marker->SetLineColor(kYellow);
      } else {
        marker->SetLineColor(kRed);
      }

      // draw 3D cross
      if (layer == 0 || fabs(layerIn - layer) == 0 || fabs(layerOut - layer) == 0) {
        marker->AddLine(doublet.first.x(),
                        doublet.first.y(),
                        doublet.first.z(),
                        doublet.second.x(),
                        doublet.second.y(),
                        doublet.second.z());
      }

      oItemHolder.AddElement(marker);
    }
  }
}

REGISTER_FWPROXYBUILDER(FWTracksterLayersProxyBuilder, ticl::Trackster, "Trackster layers", FWViewType::kISpyBit);

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
  edm::Handle<edm::ValueMap<std::pair<float, float>>> TimeValueMapHandle_;
  edm::Handle<std::vector<reco::CaloCluster>> layerClustersHandle_;
  double timeLowerBound_, timeUpperBound_;
  long layer_;
  double saturation_energy_;
  bool heatmap_;
  bool z_plus_;
  bool z_minus_;
  bool enableTimeFilter_;
  bool enablePositionLines_;
  bool enableEdges_;
  double displayMode_;
  double proportionalityFactor_;

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
  iItem->getEvent()->getByLabel(edm::InputTag("hgcalLayerClusters", "timeLayerCluster"), TimeValueMapHandle_);
  iItem->getEvent()->getByLabel(edm::InputTag("hgcalLayerClusters"), layerClustersHandle_);
  if (TimeValueMapHandle_.isValid()) {
    timeLowerBound_ = item()->getConfig()->value<double>("TimeLowerBound(ns)");
    timeUpperBound_ = item()->getConfig()->value<double>("TimeUpperBound(ns)");
    if (timeLowerBound_ > timeUpperBound_) {
      edm::LogWarning("InvalidParameters")
          << "lower time bound is larger than upper time bound. Maybe opposite is desired?";
    }
  } else {
    iItem->getEvent()->getByLabel(edm::InputTag("hgcalMergeLayerClusters", "timeLayerCluster"), TimeValueMapHandle_);
    edm::LogWarning("DataNotFound|InvalidData")
        << __FILE__ << ":" << __LINE__
        << " couldn't locate 'hgcalLayerClusters:timeLayerCluster' ValueMap in input file. Trying to access "
           "'hgcalMergeLayerClusters:timeLayerClusters' ValueMap";
    if (!TimeValueMapHandle_.isValid()) {
      edm::LogWarning("DataNotFound|InvalidData")
          << __FILE__ << ":" << __LINE__
          << " couldn't locate 'hgcalMergeLayerClusters:timeLayerCluster' ValueMap in input file.";
    }
  }

  if (!layerClustersHandle_.isValid()) {
    iItem->getEvent()->getByLabel(edm::InputTag("hgcalMergeLayerClusters"), layerClustersHandle_);
    edm::LogWarning("DataNotFound|InvalidData")
        << __FILE__ << ":" << __LINE__
        << " couldn't locate 'hgcalLayerClusters' collection "
           "in input file. Trying to access 'hgcalMergeLayerClusters' collection.";
    if (!layerClustersHandle_.isValid()) {
      edm::LogWarning("DataNotFound|InvalidData")
          << __FILE__ << ":" << __LINE__ << " couldn't locate 'hgcalMergeLayerClusters' collection in input file.";
    }
  }

  layer_ = item()->getConfig()->value<long>("Layer");
  saturation_energy_ = item()->getConfig()->value<double>("EnergyCutOff");
  heatmap_ = item()->getConfig()->value<bool>("Heatmap");
  z_plus_ = item()->getConfig()->value<bool>("Z+");
  z_minus_ = item()->getConfig()->value<bool>("Z-");
  enableTimeFilter_ = item()->getConfig()->value<bool>("EnableTimeFilter");
  enablePositionLines_ = item()->getConfig()->value<bool>("EnablePositionLines");
  enableEdges_ = item()->getConfig()->value<bool>("EnableEdges");
  displayMode_ = item()->getConfig()->value<double>("DisplayMode");
  proportionalityFactor_ = item()->getConfig()->value<double>("ProportionalityFactor");

  FWHeatmapProxyBuilderTemplate::build(iItem, product, vc);
}

void FWTracksterLayersProxyBuilder::build(const ticl::Trackster &iData,
                                          unsigned int iIndex,
                                          TEveElement &oItemHolder,
                                          const FWViewContext *) {
  if (enableTimeFilter_ && TimeValueMapHandle_.isValid()) {
    const float time = TimeValueMapHandle_->get(iIndex).first;
    if (time < timeLowerBound_ || time > timeUpperBound_)
      return;
  }

  const ticl::Trackster &trackster = iData;
  const size_t N = trackster.vertices().size();
  const std::vector<reco::CaloCluster> &layerClusters = *layerClustersHandle_;
  TEveStraightLineSet *position_marker = nullptr;

  if (enablePositionLines_) {
    position_marker = new TEveStraightLineSet;
    position_marker->SetLineWidth(2);
    position_marker->SetLineColor(kWhite);
  }

  for (size_t i = 0; i < N; ++i) {
    const reco::CaloCluster layerCluster = layerClusters[trackster.vertices(i)];
    const math::XYZPoint &position = layerCluster.position();
    const size_t nHits = layerCluster.size();
    const double energy = layerCluster.energy();
    float radius = 0;
    auto detIdOnLayer = layerCluster.seed();

    const auto *parameters = item()->getGeom()->getParameters(detIdOnLayer);
    const int layer = parameters[1];
    const int zside = parameters[2];
    const bool isSilicon = parameters[3];

    auto const z_selection_is_on = z_plus_ ^ z_minus_;
    auto const z_plus_selection_ok = z_plus_ && (zside == 1);
    auto const z_minus_selection_ok = z_minus_ && (zside == -1);
    if (!z_minus_ && !z_plus_)
      continue;
    if (z_selection_is_on && !(z_plus_selection_ok || z_minus_selection_ok))
      continue;

    if (layer_ > 0 && (layer != layer_))
      continue;

    if (displayMode_ == 0) {
      radius = sqrt(nHits);
    } else if (displayMode_ == 1) {
      radius = nHits;
    } else if (displayMode_ == 2) {
      radius = energy;
    } else if (displayMode_ == 3) {
      radius = energy / nHits;
    } else if (displayMode_ == 4) {
      float area = 0;
      if (!isSilicon) {
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

    auto *eveCircle = new TEveGeoShape("Circle");
    auto tube = new TGeoTube(0., proportionalityFactor_ * radius, 0.1);
    eveCircle->SetShape(tube);
    eveCircle->InitMainTrans();
    eveCircle->RefMainTrans().Move3PF(position.x(), position.y(), position.z());
    setupAddElement(eveCircle, &oItemHolder);
    // Apply heatmap color coding **after** the call to setupAddElement, that will internally setup the color.
    if (heatmap_) {
      const float normalized_energy = fmin(energy / saturation_energy_, 1.0f);
      const uint8_t colorFactor = gradient_steps * normalized_energy;
      eveCircle->SetFillColor(
          TColor::GetColor(gradient[0][colorFactor], gradient[1][colorFactor], gradient[2][colorFactor]));
    } else {
      eveCircle->SetMainColor(item()->modelInfo(iIndex).displayProperties().color());
      eveCircle->SetMainTransparency(item()->defaultDisplayProperties().transparency());
    }

    // seed and cluster position
    const float crossScale = 1.0f + fmin(energy, 5.0f);
    if (enablePositionLines_) {
      auto const &pos = layerCluster.position();
      const float position_crossScale = crossScale * 0.5;
      position_marker->AddLine(
          pos.x() - position_crossScale, pos.y(), pos.z(), pos.x() + position_crossScale, pos.y(), pos.z());
      position_marker->AddLine(
          pos.x(), pos.y() - position_crossScale, pos.z(), pos.x(), pos.y() + position_crossScale, pos.z());
    }
  }

  if (enablePositionLines_)
    oItemHolder.AddElement(position_marker);

  if (enableEdges_) {
    auto &edges = trackster.edges();

    TEveStraightLineSet *adjacent_marker = new TEveStraightLineSet;
    adjacent_marker->SetLineWidth(2);
    adjacent_marker->SetLineColor(kYellow);

    TEveStraightLineSet *non_adjacent_marker = new TEveStraightLineSet;
    non_adjacent_marker->SetLineWidth(2);
    non_adjacent_marker->SetLineColor(kRed);

    for (auto edge : edges) {
      auto doublet = std::make_pair(layerClusters[edge[0]], layerClusters[edge[1]]);

      int layerIn = item()->getGeom()->getParameters(doublet.first.seed())[1];
      int layerOut = item()->getGeom()->getParameters(doublet.second.seed())[1];

      const bool isAdjacent = std::abs(layerOut - layerIn) == 1;

      // draw 3D cross
      if (layer_ == 0 || fabs(layerIn - layer_) == 0 || fabs(layerOut - layer_) == 0) {
        if (isAdjacent)
          adjacent_marker->AddLine(doublet.first.x(),
                                   doublet.first.y(),
                                   doublet.first.z(),
                                   doublet.second.x(),
                                   doublet.second.y(),
                                   doublet.second.z());
        else
          non_adjacent_marker->AddLine(doublet.first.x(),
                                       doublet.first.y(),
                                       doublet.first.z(),
                                       doublet.second.x(),
                                       doublet.second.y(),
                                       doublet.second.z());
      }
    }
    oItemHolder.AddElement(adjacent_marker);
    oItemHolder.AddElement(non_adjacent_marker);
  }
}

REGISTER_FWPROXYBUILDER(FWTracksterLayersProxyBuilder, ticl::Trackster, "Trackster layers", FWViewType::kISpyBit);

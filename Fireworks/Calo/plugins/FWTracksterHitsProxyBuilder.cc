#include "Fireworks/Calo/interface/FWHeatmapProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "TEveBoxSet.h"
#include "TEveStraightLineSet.h"

class FWTracksterHitsProxyBuilder : public FWHeatmapProxyBuilderTemplate<ticl::Trackster> {
public:
  FWTracksterHitsProxyBuilder(void) {}
  ~FWTracksterHitsProxyBuilder(void) override {}

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

  FWTracksterHitsProxyBuilder(const FWTracksterHitsProxyBuilder &) = delete;                   // stop default
  const FWTracksterHitsProxyBuilder &operator=(const FWTracksterHitsProxyBuilder &) = delete;  // stop default

  void setItem(const FWEventItem *iItem) override;

  void build(const FWEventItem *iItem, TEveElementList *product, const FWViewContext *vc) override;
  void build(const ticl::Trackster &iData,
             unsigned int iIndex,
             TEveElement &oItemHolder,
             const FWViewContext *) override;
};

void FWTracksterHitsProxyBuilder::setItem(const FWEventItem *iItem) {
  FWHeatmapProxyBuilderTemplate::setItem(iItem);
  if (iItem) {
    iItem->getConfig()->assertParam("Cluster(0)/RecHit(1)", false);
    iItem->getConfig()->assertParam("EnableSeedLines", false);
    iItem->getConfig()->assertParam("EnablePositionLines", false);
    iItem->getConfig()->assertParam("EnableEdges", false);
    iItem->getConfig()->assertParam("EnableTimeFilter", false);
    iItem->getConfig()->assertParam("TimeLowerBound(ns)", 0.01, 0.0, 75.0);
    iItem->getConfig()->assertParam("TimeUpperBound(ns)", 0.01, 0.0, 75.0);
  }
}

void FWTracksterHitsProxyBuilder::build(const FWEventItem *iItem, TEveElementList *product, const FWViewContext *vc) {
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

  FWHeatmapProxyBuilderTemplate::build(iItem, product, vc);
}

void FWTracksterHitsProxyBuilder::build(const ticl::Trackster &iData,
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

  bool h_hex(false);
  TEveBoxSet *hex_boxset = new TEveBoxSet();
  if (!heatmap)
    hex_boxset->UseSingleColor();
  hex_boxset->SetPickable(true);
  hex_boxset->Reset(TEveBoxSet::kBT_Hex, true, 64);
  hex_boxset->SetAntiFlick(true);

  bool h_box(false);
  TEveBoxSet *boxset = new TEveBoxSet();
  if (!heatmap)
    boxset->UseSingleColor();
  boxset->SetPickable(true);
  boxset->Reset(TEveBoxSet::kBT_FreeBox, true, 64);
  boxset->SetAntiFlick(true);

  for (size_t i = 0; i < N; ++i) {
    const reco::CaloCluster layerCluster = layerClusters[trackster.vertices(i)];
    std::vector<std::pair<DetId, float>> clusterDetIds = layerCluster.hitsAndFractions();

    for (std::vector<std::pair<DetId, float>>::iterator it = clusterDetIds.begin(), itEnd = clusterDetIds.end();
         it != itEnd;
         ++it) {
      const uint8_t type = ((it->first >> 28) & 0xF);

      const float *corners = item()->getGeom()->getCorners(it->first);
      if (corners == nullptr)
        continue;

      if (heatmap && hitmap->find(it->first) == hitmap->end())
        continue;

      const bool z = (it->first >> 25) & 0x1;

      // discard everything thats not at the side that we are intersted in
      if (((z_plus & z_minus) != 1) && (((z_plus | z_minus) == 0) || !(z == z_minus || z == !z_plus)))
        continue;

      const float *parameters = item()->getGeom()->getParameters(it->first);
      const float *shapes = item()->getGeom()->getShapePars(it->first);

      if (parameters == nullptr || shapes == nullptr)
        continue;

      const int total_points = parameters[0];
      const bool isScintillator = (total_points == 4);

      uint8_t ll = layer;
      if (layer > 0) {
        if (layer > 28) {
          if (type == 8) {
            continue;
          }
          ll -= 28;
        } else {
          if (type != 8) {
            continue;
          }
        }

        if (ll != ((it->first >> (isScintillator ? 17 : 20)) & 0x1F))
          continue;
      }

      // seed and cluster position
      if (layerCluster.seed().rawId() == it->first.rawId()) {
        const float crossScale = 1.0f + fmin(layerCluster.energy(), 5.0f);
        if (enableSeedLines) {
          TEveStraightLineSet *marker = new TEveStraightLineSet;
          marker->SetLineWidth(1);

          // center of RecHit
          const float center[3] = {corners[total_points * 3 + 0],
                                   corners[total_points * 3 + 1],
                                   corners[total_points * 3 + 2] + shapes[3] * 0.5f};

          // draw 3D cross
          marker->AddLine(center[0] - crossScale, center[1], center[2], center[0] + crossScale, center[1], center[2]);
          marker->AddLine(center[0], center[1] - crossScale, center[2], center[0], center[1] + crossScale, center[2]);
          marker->AddLine(center[0], center[1], center[2] - crossScale, center[0], center[1], center[2] + crossScale);

          oItemHolder.AddElement(marker);
        }

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

      const float energy =
          fmin((item()->getConfig()->value<bool>("Cluster(0)/RecHit(1)") ? hitmap->at(it->first)->energy()
                                                                         : layerCluster.energy()) /
                   saturation_energy,
               1.0f);
      const uint8_t colorFactor = gradient_steps * energy;

      // Scintillator
      if (isScintillator) {
        const int total_vertices = 3 * total_points;

        std::vector<float> pnts(24);
        for (int i = 0; i < total_points; ++i) {
          pnts[i * 3 + 0] = corners[i * 3];
          pnts[i * 3 + 1] = corners[i * 3 + 1];
          pnts[i * 3 + 2] = corners[i * 3 + 2];

          pnts[(i * 3 + 0) + total_vertices] = corners[i * 3];
          pnts[(i * 3 + 1) + total_vertices] = corners[i * 3 + 1];
          pnts[(i * 3 + 2) + total_vertices] = corners[i * 3 + 2] + shapes[3];
        }
        boxset->AddBox(&pnts[0]);
        if (heatmap) {
          energy ? boxset->DigitColor(gradient[0][colorFactor], gradient[1][colorFactor], gradient[2][colorFactor])
                 : boxset->DigitColor(64, 64, 64);
        }

        h_box = true;
      }
      // Silicon
      else {
        constexpr int offset = 9;

        float centerX = (corners[6] + corners[6 + offset]) / 2;
        float centerY = (corners[7] + corners[7 + offset]) / 2;
        float radius = fabs(corners[6] - corners[6 + offset]) / 2;
        hex_boxset->AddHex(TEveVector(centerX, centerY, corners[2]), radius, 90.0, shapes[3]);
        if (heatmap) {
          energy ? hex_boxset->DigitColor(gradient[0][colorFactor], gradient[1][colorFactor], gradient[2][colorFactor])
                 : hex_boxset->DigitColor(64, 64, 64);
        }

        h_hex = true;
      }
    }
  }

  if (enableEdges) {
    auto &edges = trackster.edges();

    for (auto edge : edges) {
      auto doublet = std::make_pair(layerClusters[edge[0]], layerClusters[edge[1]]);
      TEveStraightLineSet *marker = new TEveStraightLineSet;
      marker->SetLineWidth(2);
      marker->SetLineColor(kRed);

      // draw 3D cross
      marker->AddLine(doublet.first.x(),
                      doublet.first.y(),
                      doublet.first.z(),
                      doublet.second.x(),
                      doublet.second.y(),
                      doublet.second.z());

      oItemHolder.AddElement(marker);
    }
  }

  if (h_hex) {
    hex_boxset->RefitPlex();

    hex_boxset->CSCTakeAnyParentAsMaster();
    if (!heatmap) {
      hex_boxset->CSCApplyMainColorToMatchingChildren();
      hex_boxset->CSCApplyMainTransparencyToMatchingChildren();
      hex_boxset->SetMainColor(item()->modelInfo(iIndex).displayProperties().color());
      hex_boxset->SetMainTransparency(item()->defaultDisplayProperties().transparency());
    }
    oItemHolder.AddElement(hex_boxset);
  }

  if (h_box) {
    boxset->RefitPlex();

    boxset->CSCTakeAnyParentAsMaster();
    if (!heatmap) {
      boxset->CSCApplyMainColorToMatchingChildren();
      boxset->CSCApplyMainTransparencyToMatchingChildren();
      boxset->SetMainColor(item()->modelInfo(iIndex).displayProperties().color());
      boxset->SetMainTransparency(item()->defaultDisplayProperties().transparency());
    }
    oItemHolder.AddElement(boxset);
  }
}

REGISTER_FWPROXYBUILDER(FWTracksterHitsProxyBuilder, ticl::Trackster, "Trackster hits", FWViewType::kISpyBit);

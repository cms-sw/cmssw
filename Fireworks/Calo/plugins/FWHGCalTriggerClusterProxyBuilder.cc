#include "Fireworks/Calo/plugins/FWL1THGCalProxyTemplate.cc"

#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/BuilderUtils.h"

#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "DataFormats/L1THGCal/interface/HGCalClusterT.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalTriggerDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"

#include "TEveBoxSet.h"
#include "TColor.h"
#include "TROOT.h"

class FWHGCalTriggerClusterProxyBuilder : public FWL1THGCalProxyTemplate<l1t::HGCalMulticluster> {
public:
  FWHGCalTriggerClusterProxyBuilder(void) {}
  ~FWHGCalTriggerClusterProxyBuilder(void) override {}

  REGISTER_PROXYBUILDER_METHODS();

  FWHGCalTriggerClusterProxyBuilder(const FWHGCalTriggerClusterProxyBuilder &) = delete;  // stop default
  const FWHGCalTriggerClusterProxyBuilder &operator=(const FWHGCalTriggerClusterProxyBuilder &) =
      delete;  // stop default

private:
  void build(const l1t::HGCalMulticluster &iData,
             unsigned int iIndex,
             TEveElement &oItemHolder,
             const FWViewContext *) override;
};

void FWHGCalTriggerClusterProxyBuilder::build(const l1t::HGCalMulticluster &iData,
                                              unsigned int iIndex,
                                              TEveElement &oItemHolder,
                                              const FWViewContext *) {
  const long layer = item()->getConfig()->value<long>("Layer");
  const double saturation_energy = item()->getConfig()->value<double>("EnergyCutOff");
  const bool heatmap = item()->getConfig()->value<bool>("Heatmap");

  const bool z_plus = item()->getConfig()->value<bool>("Z+");
  const bool z_minus = item()->getConfig()->value<bool>("Z-");

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

  const float energy = fmin(iData.energy() / (saturation_energy * 100), 1.0);

  for (const auto &t_detid : iData.constituents()) {
    std::unordered_set<unsigned> cells = getCellsFromTriggerCell(t_detid.first);

    for (auto &it : cells) {
      const bool z = (it >> 25) & 0x1;

      // discard everything thats not at the side that we are intersted in
      if (((z_plus & z_minus) != 1) && (((z_plus | z_minus) == 0) || !(z == z_minus || z == !z_plus)))
        continue;

      const float *corners = item()->getGeom()->getCorners(it);
      const float *parameters = item()->getGeom()->getParameters(it);
      const float *shapes = item()->getGeom()->getShapePars(it);

      if (corners == nullptr || parameters == nullptr || shapes == nullptr) {
        continue;
      }

      const int total_points = parameters[0];
      const bool isScintillator = (total_points == 4);
      const uint8_t type = ((it >> 28) & 0xF);

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

        if (ll != ((it >> (isScintillator ? 17 : 20)) & 0x1F))
          continue;
      }

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
        boxset->DigitColor(energy * 255, 0, 255 - energy * 255);

        h_box = true;
      }
      // Silicon
      else {
        const int offset = 9;

        float centerX = (corners[6] + corners[6 + offset]) / 2;
        float centerY = (corners[7] + corners[7 + offset]) / 2;
        float radius = fabs(corners[6] - corners[6 + offset]) / 2;
        hex_boxset->AddHex(TEveVector(centerX, centerY, corners[2]), radius, shapes[2], shapes[3]);
        hex_boxset->DigitColor(energy * 255, 0, 255 - energy * 255);

        h_hex = true;
      }
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

REGISTER_FWPROXYBUILDER(FWHGCalTriggerClusterProxyBuilder,
                        l1t::HGCalMulticluster,
                        "HGCal Trigger Cluster",
                        FWViewType::kAll3DBits);

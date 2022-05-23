/*
 *  FWCaloParticleProxyBuilder.cc
 *  FWorks
 *
 *  Created by Marco Rovere 13/09/2018
 *
 */

#include "Fireworks/Calo/interface/FWHeatmapProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticleFwd.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"

#include "TEveBoxSet.h"

class FWCaloParticleProxyBuilder : public FWHeatmapProxyBuilderTemplate<CaloParticle> {
public:
  FWCaloParticleProxyBuilder(void) {}
  ~FWCaloParticleProxyBuilder(void) override {}

  REGISTER_PROXYBUILDER_METHODS();

  // Disable default copy constructor
  FWCaloParticleProxyBuilder(const FWCaloParticleProxyBuilder &) = delete;
  // Disable default assignment operator
  const FWCaloParticleProxyBuilder &operator=(const FWCaloParticleProxyBuilder &) = delete;

private:
  void build(const CaloParticle &iData, unsigned int iIndex, TEveElement &oItemHolder, const FWViewContext *) override;
};

void FWCaloParticleProxyBuilder::build(const CaloParticle &iData,
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

  for (const auto &c : iData.simClusters()) {
    for (const auto &it : (*c).hits_and_fractions()) {
      if (heatmap && hitmap->find(it.first) == hitmap->end())
        continue;

      const bool z = (it.first >> 25) & 0x1;

      // discard everything thats not at the side that we are intersted in
      if (((z_plus & z_minus) != 1) && (((z_plus | z_minus) == 0) || !(z == z_minus || z == !z_plus)))
        continue;

      const float *corners = item()->getGeom()->getCorners(it.first);
      const float *parameters = item()->getGeom()->getParameters(it.first);
      const float *shapes = item()->getGeom()->getShapePars(it.first);

      if (corners == nullptr || parameters == nullptr || shapes == nullptr) {
        continue;
      }

      const int total_points = parameters[0];
      const bool isScintillator = (total_points == 4);
      const uint8_t type = ((it.first >> 28) & 0xF);

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

        if (ll != ((it.first >> (isScintillator ? 17 : 20)) & 0x1F))
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
        if (heatmap) {
          const uint8_t colorFactor = gradient_steps * (fmin(hitmap->at(it.first)->energy() / saturation_energy, 1.0f));
          boxset->DigitColor(gradient[0][colorFactor], gradient[1][colorFactor], gradient[2][colorFactor]);
        }

        h_box = true;
      }
      // Silicon
      else {
        const int offset = 9;

        float centerX = (corners[6] + corners[6 + offset]) / 2;
        float centerY = (corners[7] + corners[7 + offset]) / 2;
        float radius = fabs(corners[6] - corners[6 + offset]) / 2;
        hex_boxset->AddHex(TEveVector(centerX, centerY, corners[2]), radius, shapes[2], shapes[3]);
        if (heatmap) {
          const uint8_t colorFactor = gradient_steps * (fmin(hitmap->at(it.first)->energy() / saturation_energy, 1.0f));
          hex_boxset->DigitColor(gradient[0][colorFactor], gradient[1][colorFactor], gradient[2][colorFactor]);
        }

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

REGISTER_FWPROXYBUILDER(FWCaloParticleProxyBuilder,
                        CaloParticle,
                        "CaloParticle",
                        FWViewType::kAll3DBits | FWViewType::kAllRPZBits);

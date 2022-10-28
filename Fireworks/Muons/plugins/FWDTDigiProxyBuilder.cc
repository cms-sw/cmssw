/*
 *  FWDTDigiProxyBuilder.cc
 *  FWorks
 *
 *  Created by Ianna Osborne on 6/7/10.
 *
 */

#include "TEveGeoNode.h"
#include "TEvePointSet.h"
#include "TEveCompound.h"
#include "TGeoArb8.h"
#include "TEveBox.h"

#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/DTDigi/interface/DTDigiCollection.h"

namespace {
  void addTube(TEveBox* shape, const FWGeometry::GeomDetInfo& info, float localPos[3], const float* pars) {
    const Float_t width = pars[0] / 2.;
    const Float_t thickness = pars[1] / 2.;
    const Float_t length = pars[2] / 2.;

    const Float_t vtx[24] = {localPos[0] - width, -length, -thickness, localPos[0] - width, length,  -thickness,
                             localPos[0] + width, length,  -thickness, localPos[0] + width, -length, -thickness,
                             localPos[0] - width, -length, thickness,  localPos[0] - width, length,  thickness,
                             localPos[0] + width, length,  thickness,  localPos[0] + width, -length, thickness};

    double array[16] = {info.matrix[0],
                        info.matrix[3],
                        info.matrix[6],
                        0.,
                        info.matrix[1],
                        info.matrix[4],
                        info.matrix[7],
                        0.,
                        info.matrix[2],
                        info.matrix[5],
                        info.matrix[8],
                        0.,
                        info.translation[0],
                        info.translation[1],
                        info.translation[2],
                        1.};

    shape->SetVertices(vtx);
    shape->SetTransMatrix(array);
    shape->SetDrawFrame(false);
    shape->SetMainTransparency(75);
  }
}  // namespace

class FWDTDigiProxyBuilder : public FWProxyBuilderBase {
public:
  FWDTDigiProxyBuilder(void) {}
  ~FWDTDigiProxyBuilder(void) override {}

  bool haveSingleProduct(void) const override { return false; }

  REGISTER_PROXYBUILDER_METHODS();

  // Disable default copy constructor
  FWDTDigiProxyBuilder(const FWDTDigiProxyBuilder&) = delete;
  // Disable default assignment operator
  const FWDTDigiProxyBuilder& operator=(const FWDTDigiProxyBuilder&) = delete;

private:
  using FWProxyBuilderBase::buildViewType;
  void buildViewType(const FWEventItem* iItem,
                     TEveElementList* product,
                     FWViewType::EType,
                     const FWViewContext*) override;
};

void FWDTDigiProxyBuilder::buildViewType(const FWEventItem* iItem,
                                         TEveElementList* product,
                                         FWViewType::EType type,
                                         const FWViewContext*) {
  const DTDigiCollection* digis = nullptr;
  iItem->get(digis);

  if (!digis) {
    return;
  }
  const FWGeometry* geom = iItem->getGeom();

  for (DTDigiCollection::DigiRangeIterator dri = digis->begin(), dre = digis->end(); dri != dre; ++dri) {
    const DTLayerId& layerId = (*dri).first;
    unsigned int rawid = layerId.rawId();
    const DTDigiCollection::Range& range = (*dri).second;

    if (!geom->contains(rawid)) {
      fwLog(fwlog::kWarning) << "failed to get geometry of DT with detid: " << rawid << std::endl;

      TEveCompound* compound = createCompound();
      setupAddElement(compound, product);

      continue;
    }

    const float* pars = geom->getParameters(rawid);
    FWGeometry::IdToInfoItr det = geom->find(rawid);

    int superLayer = layerId.superlayerId().superLayer();

    float localPos[3] = {0.0, 0.0, 0.0};

    // Loop over the digis of this DetUnit
    for (DTDigiCollection::const_iterator it = range.first; it != range.second; ++it) {
      // The x wire position in the layer, starting from its wire number.
      float firstChannel = pars[3];
      float nChannels = pars[5];
      localPos[0] = ((*it).wire() - (firstChannel - 1) - 0.5) * pars[0] - nChannels / 2.0 * pars[0];

      if (type == FWViewType::k3D || type == FWViewType::kISpy) {
        TEveBox* box = new TEveBox;
        setupAddElement(box, product);
        ::addTube(box, *det, localPos, pars);
      } else if (((type == FWViewType::kRhoPhi || type == FWViewType::kRhoPhiPF) && superLayer != 2) ||
                 (type == FWViewType::kRhoZ && superLayer == 2)) {
        TEvePointSet* pointSet = new TEvePointSet;
        pointSet->SetMarkerStyle(24);
        setupAddElement(pointSet, product);

        float globalPos[3];
        geom->localToGlobal(*det, localPos, globalPos);
        pointSet->SetNextPoint(globalPos[0], globalPos[1], globalPos[2]);

        TEveBox* box = new TEveBox;
        setupAddElement(box, product);
        ::addTube(box, *det, localPos, pars);
      } else {
        TEveCompound* compound = createCompound();
        setupAddElement(compound, product);
      }
    }
  }
}

REGISTER_FWPROXYBUILDER(FWDTDigiProxyBuilder,
                        DTDigiCollection,
                        "DT Digis",
                        FWViewType::kAll3DBits | FWViewType::kAllRPZBits);

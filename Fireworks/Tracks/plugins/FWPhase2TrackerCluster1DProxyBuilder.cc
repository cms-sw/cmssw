
// -*- C++ -*-
//
// Package:     Tracks
// Class  :     FWPhase2TrackerCluster1DProxyBuilder
//
//

#include "TEveStraightLineSet.h"
#include "TEveCompound.h"
#include "TEveBox.h"
#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"

#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"

class FWPhase2TrackerCluster1DProxyBuilder : public FWProxyBuilderBase {
public:
  FWPhase2TrackerCluster1DProxyBuilder(void) {}
  ~FWPhase2TrackerCluster1DProxyBuilder(void) override {}

  REGISTER_PROXYBUILDER_METHODS();

  FWPhase2TrackerCluster1DProxyBuilder(const FWPhase2TrackerCluster1DProxyBuilder&) = delete;
  const FWPhase2TrackerCluster1DProxyBuilder& operator=(const FWPhase2TrackerCluster1DProxyBuilder&) = delete;

private:
  void localModelChanges(const FWModelId& iId,
                         TEveElement* parent,
                         FWViewType::EType viewType,
                         const FWViewContext* vc) override;

  using FWProxyBuilderBase::build;
  void build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*) override;
};

void FWPhase2TrackerCluster1DProxyBuilder::build(const FWEventItem* iItem,
                                                 TEveElementList* product,
                                                 const FWViewContext*) {
  const Phase2TrackerCluster1DCollectionNew* pixels = nullptr;

  iItem->get(pixels);

  if (!pixels) {
    fwLog(fwlog::kWarning) << "failed get SiPixelDigis" << std::endl;
    return;
  }

  for (Phase2TrackerCluster1DCollectionNew::const_iterator set = pixels->begin(), setEnd = pixels->end(); set != setEnd;
       ++set) {
    unsigned int id = set->detId();

    const FWGeometry* geom = iItem->getGeom();
    const float* pars = geom->getParameters(id);
    const float* shape = geom->getShapePars(id);

    const edmNew::DetSet<Phase2TrackerCluster1D>& clusters = *set;

    for (edmNew::DetSet<Phase2TrackerCluster1D>::const_iterator itc = clusters.begin(), edc = clusters.end();
         itc != edc;
         ++itc) {
      TEveElement* itemHolder = createCompound();
      product->AddElement(itemHolder);

      if (!geom->contains(id)) {
        fwLog(fwlog::kWarning) << "failed get geometry of Phase2TrackerCluster1D with detid: " << id << std::endl;
        continue;
      }

      float halfLength = shape[2];
      float pitchSecond = pars[1];

      // line
      //
      TEveStraightLineSet* lineSet = new TEveStraightLineSet;
      float localPointBeg[3] = {fireworks::phase2PixelLocalX((*itc).center(), pars, shape),
                                float((*itc).column()) * pitchSecond - halfLength,
                                0.0};
      float localPointEnd[3] = {fireworks::phase2PixelLocalX((*itc).center(), pars, shape),
                                float((*itc).column() + 1.0) * pitchSecond - halfLength,
                                0.0};

      float globalPointBeg[3];
      float globalPointEnd[3];
      geom->localToGlobal(id, localPointBeg, globalPointBeg);
      geom->localToGlobal(id, localPointEnd, globalPointEnd);

      lineSet->AddLine(globalPointBeg, globalPointEnd);
      lineSet->AddMarker(0, 0.5f);

      const FWDisplayProperties& dp = FWProxyBuilderBase::item()->defaultDisplayProperties();
      lineSet->SetMarkerColor(dp.color());

      setupAddElement(lineSet, itemHolder);
    }
  }
}

void FWPhase2TrackerCluster1DProxyBuilder::localModelChanges(const FWModelId& iId,
                                                             TEveElement* parent,
                                                             FWViewType::EType viewType,
                                                             const FWViewContext* vc) {
  if (TEveStraightLineSet* ls = dynamic_cast<TEveStraightLineSet*>(*parent->BeginChildren())) {
    Color_t c = FWProxyBuilderBase::item()->modelInfo(iId.index()).displayProperties().color();
    for (TEveProjectable::ProjList_i j = ls->BeginProjecteds(); j != ls->EndProjecteds(); ++j) {
      if (TEveStraightLineSet* pls = dynamic_cast<TEveStraightLineSet*>(*j)) {
        pls->SetMarkerColor(c);
        pls->ElementChanged();
      }
    }

    ls->SetMarkerColor(c);
    ls->ElementChanged();
  }
}

REGISTER_FWPROXYBUILDER(FWPhase2TrackerCluster1DProxyBuilder,
                        Phase2TrackerCluster1DCollectionNew,
                        "Phase2TrackerCluster1D",
                        FWViewType::kAll3DBits | FWViewType::kAllRPZBits);

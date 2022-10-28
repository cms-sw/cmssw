// -*- C++ -*-
//
// Package:     Tracks
// Class  :     FWSiPixelDigiProxyBuilder
//
//
// Original Author:
//         Created:  Thu Dec  6 18:01:21 PST 2007
//

#include "TEveCompound.h"
#include "TEvePointSet.h"

#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"

#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/Common/interface/DetSetVector.h"

class FWSiPixelDigiProxyBuilder : public FWProxyBuilderBase {
public:
  FWSiPixelDigiProxyBuilder(void) {}
  ~FWSiPixelDigiProxyBuilder(void) override {}

  REGISTER_PROXYBUILDER_METHODS();

  // Disable default copy constructor
  FWSiPixelDigiProxyBuilder(const FWSiPixelDigiProxyBuilder&) = delete;
  // Disable default assignment operator
  const FWSiPixelDigiProxyBuilder& operator=(const FWSiPixelDigiProxyBuilder&) = delete;

private:
  using FWProxyBuilderBase::build;
  void build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*) override;
};

void FWSiPixelDigiProxyBuilder::build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*) {
  const edm::DetSetVector<PixelDigi>* digis = nullptr;
  iItem->get(digis);

  if (!digis) {
    return;
  }
  const FWGeometry* geom = iItem->getGeom();

  for (edm::DetSetVector<PixelDigi>::const_iterator it = digis->begin(), end = digis->end(); it != end; ++it) {
    edm::DetSet<PixelDigi> ds = *it;
    unsigned int id = ds.id;

    const float* pars = geom->getParameters(id);

    for (edm::DetSet<PixelDigi>::const_iterator idigi = ds.data.begin(), idigiEnd = ds.data.end(); idigi != idigiEnd;
         ++idigi) {
      TEvePointSet* pointSet = new TEvePointSet;
      pointSet->SetMarkerSize(2);
      pointSet->SetMarkerStyle(2);
      setupAddElement(pointSet, product);

      if (!geom->contains(id)) {
        fwLog(fwlog::kWarning) << "failed get geometry of SiPixelDigi with detid: " << id << std::endl;
      } else {
        float localPoint[3] = {
            fireworks::pixelLocalX((*idigi).row(), pars), fireworks::pixelLocalY((*idigi).column(), pars), 0.0};

        float globalPoint[3];
        geom->localToGlobal(id, localPoint, globalPoint);

        pointSet->SetNextPoint(globalPoint[0], globalPoint[1], globalPoint[2]);
      }
    }  // end of iteration over digis in range
  }    // end of iteration over the DetSetVector
}

REGISTER_FWPROXYBUILDER(FWSiPixelDigiProxyBuilder,
                        edm::DetSetVector<PixelDigi>,
                        "SiPixelDigi",
                        FWViewType::kAll3DBits | FWViewType::kAllRPZBits);

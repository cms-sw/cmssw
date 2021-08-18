#ifndef RecoParticleFlow_PFClusterProducer_PFHcalRecHitCreator_h
#define RecoParticleFlow_PFClusterProducer_PFHcalRecHitCreator_h

#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitCreatorBase.h"

#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"

template <typename Digi, typename Geometry, PFLayer::Layer Layer, int Detector>
class PFHcalRecHitCreator final : public PFRecHitCreatorBase {
public:
  PFHcalRecHitCreator(const edm::ParameterSet& iConfig, edm::ConsumesCollector& cc)
      : PFRecHitCreatorBase(iConfig, cc),
        recHitToken_(cc.consumes<edm::SortedCollection<Digi> >(iConfig.getParameter<edm::InputTag>("src"))),
        geomToken_(cc.esConsumes()),
        topoToken_(cc.esConsumes()) {}

  void importRecHits(std::unique_ptr<reco::PFRecHitCollection>& out,
                     std::unique_ptr<reco::PFRecHitCollection>& cleaned,
                     const edm::Event& iEvent,
                     const edm::EventSetup& iSetup) override {
    beginEvent(iEvent, iSetup);

    edm::Handle<edm::SortedCollection<Digi> > recHitHandle;

    edm::ESHandle<CaloGeometry> geoHandle = iSetup.getHandle(geomToken_);
    edm::ESHandle<HcalTopology> hcalTopology = iSetup.getHandle(topoToken_);

    // get the hcal geometry and topology
    const CaloSubdetectorGeometry* gTmp = geoHandle->getSubdetectorGeometry(DetId::Hcal, Detector);
    const Geometry* hcalGeo = dynamic_cast<const Geometry*>(gTmp);
    const HcalTopology* theHcalTopology = hcalTopology.product();

    iEvent.getByToken(recHitToken_, recHitHandle);
    for (const auto& erh : *recHitHandle) {
      HcalDetId detid = (HcalDetId)erh.detid();
      HcalSubdetector esd = (HcalSubdetector)detid.subdetId();

      //since hbhe are together kill other detector
      if (esd != Detector && Detector != HcalOther)
        continue;

      if (theHcalTopology->getMergePositionFlag() && esd == HcalEndcap) {
        detid = theHcalTopology->idFront(detid);
      }

      auto energy = erh.energy();
      auto time = erh.time();
      auto depth = detid.depth();

      auto thisCell = hcalGeo->getGeometry(detid);

      // find rechit geometry
      if (!thisCell) {
        edm::LogError("PFHcalRecHitCreator")
            << "warning detid " << detid.rawId() << " not found in geometry" << std::endl;
        continue;
      }

      reco::PFRecHit rh(thisCell, detid.rawId(), Layer, energy);
      rh.setTime(time);  //Mike: This we will use later
      rh.setDepth(depth);

      bool rcleaned = false;
      bool keep = true;

      //Apply Q tests
      for (const auto& qtest : qualityTests_) {
        if (!qtest->test(rh, erh, rcleaned)) {
          keep = false;
        }
      }

      if (keep) {
        out->push_back(std::move(rh));
      } else if (rcleaned)
        cleaned->push_back(std::move(rh));
    }
  }

protected:
  edm::EDGetTokenT<edm::SortedCollection<Digi> > recHitToken_;
  int hoDepth_;

private:
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> topoToken_;
};

typedef PFHcalRecHitCreator<HBHERecHit, CaloSubdetectorGeometry, PFLayer::HCAL_BARREL1, HcalBarrel> PFHBRecHitCreator;
typedef PFHcalRecHitCreator<HORecHit, CaloSubdetectorGeometry, PFLayer::HCAL_BARREL2, HcalOuter> PFHORecHitCreator;
typedef PFHcalRecHitCreator<HBHERecHit, CaloSubdetectorGeometry, PFLayer::HCAL_ENDCAP, HcalEndcap> PFHERecHitCreator;
typedef PFHcalRecHitCreator<HFRecHit, CaloSubdetectorGeometry, PFLayer::HF_EM, HcalForward> PFHFEMRecHitCreator;
typedef PFHcalRecHitCreator<HFRecHit, CaloSubdetectorGeometry, PFLayer::HF_HAD, HcalForward> PFHFHADRecHitCreator;

#endif

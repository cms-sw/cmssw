#ifndef RecoParticleFlow_PFClusterProducer_PFHGCalRecHitCreator_h
#define RecoParticleFlow_PFClusterProducer_PFHGCalRecHitCreator_h

#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitCreatorBase.h"

#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

template <typename DET, PFLayer::Layer Layer, DetId::Detector det, unsigned subdet>
class PFHGCalRecHitCreator : public PFRecHitCreatorBase {
public:
  PFHGCalRecHitCreator(const edm::ParameterSet& iConfig, edm::ConsumesCollector& cc)
      : PFRecHitCreatorBase(iConfig, cc),
        recHitToken_(cc.consumes<HGCRecHitCollection>(iConfig.getParameter<edm::InputTag>("src"))),
        geometryInstance_(iConfig.getParameter<std::string>("geometryInstance")),
        geomToken_(cc.esConsumes()) {}

  void importRecHits(std::unique_ptr<reco::PFRecHitCollection>& out,
                     std::unique_ptr<reco::PFRecHitCollection>& cleaned,
                     const edm::Event& iEvent,
                     const edm::EventSetup& iSetup) override {
    // Setup RecHitTools to properly compute the position of the HGCAL Cells vie their DetIds
    edm::ESHandle<CaloGeometry> geoHandle = iSetup.getHandle(geomToken_);
    recHitTools_.setGeometry(*geoHandle);

    for (unsigned int i = 0; i < qualityTests_.size(); ++i) {
      qualityTests_.at(i)->beginEvent(iEvent, iSetup);
    }

    edm::Handle<HGCRecHitCollection> recHitHandle;
    iEvent.getByToken(recHitToken_, recHitHandle);
    const HGCRecHitCollection& rechits = *recHitHandle;

    const CaloGeometry* geom = geoHandle.product();

    unsigned skipped_rechits = 0;
    for (const auto& hgrh : rechits) {
      const DET detid(hgrh.detid());

      if (det != detid.det() or (subdet != 0 and subdet != detid.subdetId())) {
        throw cms::Exception("IncorrectHGCSubdetector")
            << "det expected: " << det << " det gotten: " << detid.det() << " ; "
            << "subdet expected: " << subdet << " subdet gotten: " << detid.subdetId() << std::endl;
      }

      double energy = hgrh.energy();
      double time = hgrh.time();

      auto thisCell = geom->getSubdetectorGeometry(det, subdet)->getGeometry(detid);

      // find rechit geometry
      if (!thisCell) {
        LogDebug("PFHGCalRecHitCreator") << "warning detid " << detid.rawId() << " not found in geometry" << std::endl;
        ++skipped_rechits;
        continue;
      }

      reco::PFRecHit rh(thisCell, detid.rawId(), Layer, energy);

      bool rcleaned = false;
      bool keep = true;

      //Apply Q tests
      for (unsigned int i = 0; i < qualityTests_.size(); ++i) {
        if (!qualityTests_.at(i)->test(rh, hgrh, rcleaned)) {
          keep = false;
        }
      }

      if (keep) {
        rh.setTime(time);
        out->push_back(rh);
      } else if (rcleaned)
        cleaned->push_back(rh);
    }
    edm::LogInfo("HGCalRecHitCreator") << "Skipped " << skipped_rechits << " out of " << rechits.size() << " rechits!"
                                       << std::endl;
    edm::LogInfo("HGCalRecHitCreator") << "Created " << out->size() << " PFRecHits!" << std::endl;
  }

protected:
  edm::EDGetTokenT<HGCRecHitCollection> recHitToken_;
  std::string geometryInstance_;

private:
  hgcal::RecHitTools recHitTools_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
};

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"

typedef PFHGCalRecHitCreator<HGCalDetId, PFLayer::HGCAL, DetId::Forward, HGCEE> PFHGCEERecHitCreator;
typedef PFHGCalRecHitCreator<HGCalDetId, PFLayer::HGCAL, DetId::Forward, HGCHEF> PFHGCHEFRecHitCreator;
typedef PFHGCalRecHitCreator<HcalDetId, PFLayer::HGCAL, DetId::Hcal, HcalEndcap> PFHGCHEBRecHitCreator;

typedef PFHGCalRecHitCreator<HGCSiliconDetId, PFLayer::HGCAL, DetId::HGCalEE, ForwardEmpty> PFHGCalEERecHitCreator;
typedef PFHGCalRecHitCreator<HGCSiliconDetId, PFLayer::HGCAL, DetId::HGCalHSi, ForwardEmpty> PFHGCalHSiRecHitCreator;
typedef PFHGCalRecHitCreator<HGCScintillatorDetId, PFLayer::HGCAL, DetId::HGCalHSc, ForwardEmpty>
    PFHGCalHScRecHitCreator;

#endif

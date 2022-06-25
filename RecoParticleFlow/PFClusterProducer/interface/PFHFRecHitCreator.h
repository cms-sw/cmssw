#ifndef RecoParticleFlow_PFClusterProducer_PFHFRecHitCreator_h
#define RecoParticleFlow_PFClusterProducer_PFHFRecHitCreator_h

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
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloGeometry/interface/IdealZPrism.h"

#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"

class PFHFRecHitCreator final : public PFRecHitCreatorBase {
public:
  PFHFRecHitCreator(const edm::ParameterSet& iConfig, edm::ConsumesCollector& cc)
      : PFRecHitCreatorBase(iConfig, cc),
        recHitToken_(cc.consumes<edm::SortedCollection<HFRecHit> >(iConfig.getParameter<edm::InputTag>("src"))),
        EM_Depth_(iConfig.getParameter<double>("EMDepthCorrection")),
        HAD_Depth_(iConfig.getParameter<double>("HADDepthCorrection")),
        shortFibre_Cut(iConfig.getParameter<double>("ShortFibre_Cut")),
        longFibre_Fraction(iConfig.getParameter<double>("LongFibre_Fraction")),
        longFibre_Cut(iConfig.getParameter<double>("LongFibre_Cut")),
        shortFibre_Fraction(iConfig.getParameter<double>("ShortFibre_Fraction")),
        thresh_HF_(iConfig.getParameter<double>("thresh_HF")),
        HFCalib_(iConfig.getParameter<double>("HFCalib29")),
        geomToken_(cc.esConsumes()) {}

  void importRecHits(std::unique_ptr<reco::PFRecHitCollection>& out,
                     std::unique_ptr<reco::PFRecHitCollection>& cleaned,
                     const edm::Event& iEvent,
                     const edm::EventSetup& iSetup) override {
    reco::PFRecHitCollection tmpOut;

    beginEvent(iEvent, iSetup);

    edm::Handle<edm::SortedCollection<HFRecHit> > recHitHandle;

    edm::ESHandle<CaloGeometry> geoHandle = iSetup.getHandle(geomToken_);

    // get the ecal geometry
    const CaloSubdetectorGeometry* hcalGeo = geoHandle->getSubdetectorGeometry(DetId::Hcal, HcalForward);

    iEvent.getByToken(recHitToken_, recHitHandle);
    for (const auto& erh : *recHitHandle) {
      const HcalDetId& detid = (HcalDetId)erh.detid();
      auto depth = detid.depth();

      // ATTN: skip dual anode in HF for now (should be fixed in upstream changes)
      if (depth > 2)
        continue;

      auto energy = erh.energy();
      auto time = erh.time();

      std::shared_ptr<const CaloCellGeometry> thisCell = hcalGeo->getGeometry(detid);
      auto zp = dynamic_cast<IdealZPrism const*>(thisCell.get());
      assert(zp);
      thisCell = zp->forPF();

      // find rechit geometry
      if (!thisCell) {
        edm::LogError("PFHFRecHitCreator")
            << "warning detid " << detid.rawId() << " not found in geometry" << std::endl;
        continue;
      }

      PFLayer::Layer layer = depth == 1 ? PFLayer::HF_EM : PFLayer::HF_HAD;

      reco::PFRecHit rh(thisCell, detid.rawId(), layer, energy);
      rh.setTime(time);
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
        tmpOut.push_back(std::move(rh));
      } else if (rcleaned)
        cleaned->push_back(std::move(rh));
    }
    //Sort by DetID the collection
    DetIDSorter sorter;
    if (!tmpOut.empty())
      std::sort(tmpOut.begin(), tmpOut.end(), sorter);

    /////////////////////HF DUAL READOUT/////////////////////////

    for (auto& hit : tmpOut) {
      reco::PFRecHit newHit = hit;
      const HcalDetId& detid = (HcalDetId)hit.detId();
      if (detid.depth() == 1) {
        double lONG = hit.energy();
        //find the short hit
        HcalDetId shortID(HcalForward, detid.ieta(), detid.iphi(), 2);
        auto found_hit =
            std::lower_bound(tmpOut.begin(), tmpOut.end(), shortID, [](const reco::PFRecHit& a, HcalDetId b) {
              return a.detId() < b.rawId();
            });
        if (found_hit != tmpOut.end() && found_hit->detId() == shortID.rawId()) {
          double sHORT = found_hit->energy();
          //Ask for fraction
          double energy = lONG - sHORT;

          if (abs(detid.ieta()) <= 32)
            energy *= HFCalib_;
          newHit.setEnergy(energy);
          if (!(lONG > longFibre_Cut && (sHORT / lONG < shortFibre_Fraction)))
            if (energy > thresh_HF_)
              out->push_back(newHit);
        } else {
          //make only long hit
          double energy = lONG;
          if (abs(detid.ieta()) <= 32)
            energy *= HFCalib_;
          newHit.setEnergy(energy);

          if (energy > thresh_HF_)
            out->push_back(newHit);
        }

      } else {
        double sHORT = hit.energy();
        HcalDetId longID(HcalForward, detid.ieta(), detid.iphi(), 1);
        auto found_hit =
            std::lower_bound(tmpOut.begin(), tmpOut.end(), longID, [](const reco::PFRecHit& a, HcalDetId b) {
              return a.detId() < b.rawId();
            });
        double energy = sHORT;
        if (found_hit != tmpOut.end() && found_hit->detId() == longID.rawId()) {
          double lONG = found_hit->energy();
          //Ask for fraction

          //If in this case lONG-sHORT<0 add the energy to the sHORT
          if ((lONG - sHORT) < thresh_HF_)
            energy += lONG;
          else
            energy += sHORT;

          if (abs(detid.ieta()) <= 32)
            energy *= HFCalib_;

          newHit.setEnergy(energy);
          if (!(sHORT > shortFibre_Cut && (lONG / sHORT < longFibre_Fraction)))
            if (energy > thresh_HF_)
              out->push_back(newHit);

        } else {
          //only short hit!
          if (abs(detid.ieta()) <= 32)
            energy *= HFCalib_;
          newHit.setEnergy(energy);
          if (energy > thresh_HF_)
            out->push_back(newHit);
        }
      }
    }
  }

protected:
  edm::EDGetTokenT<edm::SortedCollection<HFRecHit> > recHitToken_;
  double EM_Depth_;
  double HAD_Depth_;
  // Don't allow large energy in short fibres if there is no energy in long fibres
  double shortFibre_Cut;
  double longFibre_Fraction;

  // Don't allow large energy in long fibres if there is no energy in short fibres
  double longFibre_Cut;
  double shortFibre_Fraction;
  double thresh_HF_;
  double HFCalib_;

  class DetIDSorter {
  public:
    DetIDSorter() = default;
    ~DetIDSorter() = default;

    bool operator()(const reco::PFRecHit& a, const reco::PFRecHit& b) {
      if (DetId(a.detId()).det() == DetId::Hcal || DetId(b.detId()).det() == DetId::Hcal)
        return (HcalDetId)(a.detId()) < (HcalDetId)(b.detId());
      else
        return a.detId() < b.detId();
    }
  };

private:
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
};
#endif

#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/HGCalReco/interface/HGCalSoARecHitsHostCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoARecHitsDeviceCollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class HGCalSoARecHitsProducer : public stream::EDProducer<> {
  public:
    HGCalSoARecHitsProducer(edm::ParameterSet const& config)
        : EDProducer(config),
          detector_(config.getParameter<std::string>("detector")),
          initialized_(false),
          isNose_(detector_ == "HFNose"),
          maxNumberOfThickIndices_(config.getParameter<unsigned>("maxNumberOfThickIndices")),
          fcPerEle_(config.getParameter<double>("fcPerEle")),
          ecut_(config.getParameter<double>("ecut")),
          fcPerMip_(config.getParameter<std::vector<double>>("fcPerMip")),
          nonAgedNoises_(config.getParameter<std::vector<double>>("noises")),
          dEdXweights_(config.getParameter<std::vector<double>>("dEdXweights")),
          thicknessCorrection_(config.getParameter<std::vector<double>>("thicknessCorrection")),
          caloGeomToken_(consumesCollector().esConsumes<CaloGeometry, CaloGeometryRecord>()),
          hits_token_(consumes<HGCRecHitCollection>(config.getParameter<edm::InputTag>("recHits"))),
          deviceToken_{produces()} {}

    ~HGCalSoARecHitsProducer() override = default;

    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override {
      edm::Handle<HGCRecHitCollection> hits_h;

      edm::ESHandle<CaloGeometry> geom = iSetup.getHandle(caloGeomToken_);
      rhtools_.setGeometry(*geom);
      maxlayer_ = rhtools_.lastLayer(isNose_);

      hits_h = iEvent.getHandle(hits_token_);
      auto const& hits = *(hits_h.product());
      computeThreshold();

      // Count effective hits above threshold
      uint32_t index = 0;
      for (unsigned int i = 0; i < hits.size(); ++i) {
        const HGCRecHit& hgrh = hits[i];
        DetId detid = hgrh.detid();
        unsigned int layerOnSide = (rhtools_.getLayerWithOffset(detid) - 1);

        // set sigmaNoise default value 1 to use kappa value directly in case of
        // sensor-independent thresholds
        int thickness_index = rhtools_.getSiThickIndex(detid);
        if (thickness_index == -1) {
          thickness_index = maxNumberOfThickIndices_;
        }
        double storedThreshold = thresholds_[layerOnSide][thickness_index];
        if (hgrh.energy() < storedThreshold)
          continue;  // this sets the ZS threshold at ecut times the sigma noise
        index++;
      }

      // Allocate Host SoA will contain one entry for each RecHit above threshold
      HGCalSoARecHitsHostCollection cells(index, iEvent.queue());
      auto cellsView = cells.view();

      // loop over all hits and create the Hexel structure, skip energies below ecut
      // for each layer and wafer calculate the thresholds (sigmaNoise and energy)
      // once
      index = 0;
      for (unsigned int i = 0; i < hits.size(); ++i) {
        const HGCRecHit& hgrh = hits[i];
        DetId detid = hgrh.detid();
        unsigned int layerOnSide = (rhtools_.getLayerWithOffset(detid) - 1);

        // set sigmaNoise default value 1 to use kappa value directly in case of
        // sensor-independent thresholds
        float sigmaNoise = 1.f;
        int thickness_index = rhtools_.getSiThickIndex(detid);
        if (thickness_index == -1) {
          thickness_index = maxNumberOfThickIndices_;
        }
        double storedThreshold = thresholds_[layerOnSide][thickness_index];
        if (detid.det() == DetId::HGCalHSi || detid.subdetId() == HGCHEF) {
          storedThreshold = thresholds_.at(layerOnSide).at(thickness_index + deltasi_index_regemfac_);
        }
        sigmaNoise = v_sigmaNoise_.at(layerOnSide).at(thickness_index);

        if (hgrh.energy() < storedThreshold)
          continue;  // this sets the ZS threshold at ecut times the sigma noise
        // for the sensor

        const GlobalPoint position(rhtools_.getPosition(detid));
        int offset = ((rhtools_.zside(detid) + 1) >> 1) * maxlayer_;
        int layer = layerOnSide + offset;
        auto entryInSoA = cellsView[index];
        if (detector_ == "BH") {
          entryInSoA.dim1() = position.eta();
          entryInSoA.dim2() = position.phi();
        }  // else, isSilicon == true and eta phi values will not be used
        else {
          entryInSoA.dim1() = position.x();
          entryInSoA.dim2() = position.y();
        }
        entryInSoA.dim3() = position.z();
        entryInSoA.weight() = hgrh.energy();
        entryInSoA.sigmaNoise() = sigmaNoise;
        entryInSoA.layer() = layer;
        entryInSoA.recHitIndex() = i;
        entryInSoA.detid() = detid.rawId();
        entryInSoA.time() = hgrh.time();
        entryInSoA.timeError() = hgrh.timeError();
        index++;
      }
#if 0
        std::cout << "Size: " << cells->metadata().size() << " count cells: " << index
          << " i.e. " << cells->metadata().size() << std::endl;
#endif

      if constexpr (!std::is_same_v<Device, alpaka_common::DevHost>) {
        // Trigger copy async to GPU
        //std::cout << "GPU" << std::endl;
        HGCalSoARecHitsDeviceCollection deviceProduct{cells->metadata().size(), iEvent.queue()};
        alpaka::memcpy(iEvent.queue(), deviceProduct.buffer(), cells.const_buffer());
        iEvent.emplace(deviceToken_, std::move(deviceProduct));
      } else {
        //std::cout << "CPU" << std::endl;
        iEvent.emplace(deviceToken_, std::move(cells));
      }
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<std::string>("detector", "EE")->setComment("options EE, FH, BH,  HFNose; other value defaults to EE");
      desc.add<edm::InputTag>("recHits", edm::InputTag("HGCalRecHit", "HGCEERecHits"));
      desc.add<unsigned int>("maxNumberOfThickIndices", 6);
      desc.add<double>("fcPerEle", 0.00016020506);
      desc.add<std::vector<double>>("fcPerMip");
      desc.add<std::vector<double>>("thicknessCorrection");
      desc.add<std::vector<double>>("noises");
      desc.add<std::vector<double>>("dEdXweights");
      desc.add<double>("ecut", 3.);
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    std::string detector_;
    bool initialized_;
    bool isNose_;
    unsigned maxNumberOfThickIndices_;
    unsigned int maxlayer_;
    int deltasi_index_regemfac_;
    double sciThicknessCorrection_;
    double fcPerEle_;
    double ecut_;
    std::vector<double> fcPerMip_;
    std::vector<double> nonAgedNoises_;
    std::vector<double> dEdXweights_;
    std::vector<double> thicknessCorrection_;
    std::vector<std::vector<double>> thresholds_;
    std::vector<std::vector<double>> v_sigmaNoise_;

    hgcal::RecHitTools rhtools_;
    edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeomToken_;
    edm::EDGetTokenT<HGCRecHitCollection> hits_token_;
    device::EDPutToken<HGCalSoARecHitsDeviceCollection> const deviceToken_;

    void computeThreshold() {
      // To support the TDR geometry and also the post-TDR one (v9 onwards), we
      // need to change the logic of the vectors containing signal to noise and
      // thresholds. The first 3 indices will keep on addressing the different
      // thicknesses of the Silicon detectors in CE_E , the next 3 indices will address
      // the thicknesses of the Silicon detectors in CE_H, while the last one, number 6 (the
      // seventh) will address the Scintillators. This change will support both
      // geometries at the same time.

      if (initialized_)
        return;  // only need to calculate thresholds once

      initialized_ = true;

      std::vector<double> dummy;

      dummy.resize(maxNumberOfThickIndices_ + !isNose_, 0);  // +1 to accomodate for the Scintillators
      thresholds_.resize(maxlayer_, dummy);
      v_sigmaNoise_.resize(maxlayer_, dummy);

      for (unsigned ilayer = 1; ilayer <= maxlayer_; ++ilayer) {
        for (unsigned ithick = 0; ithick < maxNumberOfThickIndices_; ++ithick) {
          float sigmaNoise = 0.001f * fcPerEle_ * nonAgedNoises_[ithick] * dEdXweights_[ilayer] /
                             (fcPerMip_[ithick] * thicknessCorrection_[ithick]);
          thresholds_[ilayer - 1][ithick] = sigmaNoise * ecut_;
          v_sigmaNoise_[ilayer - 1][ithick] = sigmaNoise;
#if 0
            std::cout << "ilayer: " << ilayer << " nonAgedNoises: " << nonAgedNoises_[ithick]
              << " fcPerEle: " << fcPerEle_ << " fcPerMip: " << fcPerMip_[ithick]
              << " noiseMip: " << fcPerEle_ * nonAgedNoises_[ithick] / fcPerMip_[ithick]
              << " sigmaNoise: " << sigmaNoise << "\n";
#endif
        }
      }
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(HGCalSoARecHitsProducer);

#include <cmath>
#include <cstdint>
#include <vector>

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
#include "RecoLocalCalo/HGCalRecAlgos/interface/TICLGeomTools.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class HGCalSoARecHitsProducer : public stream::EDProducer<> {
  public:
    HGCalSoARecHitsProducer(edm::ParameterSet const& config)
        : EDProducer(config),
          detector_(config.getParameter<std::string>("detector")),
          initialized_(false),
          isNose_(detector_ == "HFNose"),
          maxNumberOfThickIndices_(config.getParameter<unsigned>("maxNumberOfThickIndices")),
          fcPerEle_(config.getParameter<float>("fcPerEle")),
          ecut_(config.getParameter<float>("ecut")),
          fcPerMip_(config.getParameter<std::vector<float>>("fcPerMip")),
          nonAgedNoises_(config.getParameter<std::vector<float>>("noises")),
          dEdXweights_(config.getParameter<std::vector<float>>("dEdXweights")),
          thicknessCorrection_(config.getParameter<std::vector<float>>("thicknessCorrection")),
          noiseMip_(config.getParameter<double>("noiseMip")),
          sciThicknessCorrection_(config.getParameter<double>("sciThicknessCorrection")),
          ticlGeomToken_(consumesCollector().esConsumes<TICLGeomHost, CaloGeometryRecord>(edm::ESInputTag("", ""))),
          ticlGeomLookupToken_(
              consumesCollector().esConsumes<TICLGeomLookupHost, CaloGeometryRecord>(edm::ESInputTag("", ""))),
          ticlGeomLayersToken_(
              consumesCollector().esConsumes<TICLGeomLayersHost, CaloGeometryRecord>(edm::ESInputTag("", ""))),
          hits_token_(consumes<HGCRecHitCollection>(config.getParameter<edm::InputTag>("recHits"))),
          deviceToken_{produces()},
          layerSizesToken_{produces("layerSizes")} {
      // Offset to jump from the CE-E silicon thickness indices to the CE-H ones
      // in the thresholds array. It equals the number of CE-E silicon thickness
      // categories, (half of the total number of silicon thickness indices)
      // (3 for the pre-v19 geometries, 4 for v19).
      deltasi_index_regemfac_ = maxNumberOfThickIndices_ / 2;
    }

    ~HGCalSoARecHitsProducer() override = default;

    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override {
      edm::Handle<HGCRecHitCollection> hits_h;

      rhtools_.setGeometry(
          iSetup.getData(ticlGeomToken_), iSetup.getData(ticlGeomLookupToken_), iSetup.getData(ticlGeomLayersToken_));
      maxlayer_ = rhtools_.lastLayer(isNose_);

      hits_h = iEvent.getHandle(hits_token_);
      auto const& hits = *(hits_h.product());
      computeThreshold();

      // The rechit SoA is emitted layer-contiguous: hits are grouped by their
      // global layer index (layerOnSide + zside * maxlayer_), so both endcaps
      // together span 2 * maxlayer_ layer slots.
      const unsigned int numberOfLayers = 2 * maxlayer_;

      std::vector<uint32_t> hitsPerLayer(numberOfLayers, 0);
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
        if (detid.det() == DetId::HGCalHSi || detid.subdetId() == HGCHEF) {
          storedThreshold = thresholds_.at(layerOnSide).at(thickness_index + deltasi_index_regemfac_);
        }
        if (hgrh.energy() < storedThreshold)
          continue;  // this sets the ZS threshold at ecut times the sigma noise
        const int offset = ((rhtools_.zside(detid) + 1) >> 1) * maxlayer_;
        const int layer = layerOnSide + offset;
        hitsPerLayer[layer]++;
        index++;
      }

      // Allocate Host SoA will contain one entry for each RecHit above threshold
      HGCalSoARecHitsHostCollection cells(iEvent.queue(), index);
      auto cellsView = cells.view();

      std::vector<uint32_t> layerCursor(numberOfLayers, 0);
      std::vector<uint32_t> layerSizes;
      layerSizes.reserve(numberOfLayers);
      uint32_t nextLayerStart = 0;
      for (unsigned int l = 0; l < numberOfLayers; ++l) {
        layerCursor[l] = nextLayerStart;
        nextLayerStart += hitsPerLayer[l];
        if (hitsPerLayer[l] > 0)
          layerSizes.push_back(hitsPerLayer[l]);
      }

      // loop over all hits and create the Hexel structure, skip energies below ecut
      // for each layer and wafer calculate the thresholds (sigmaNoise and energy)
      // once. Hits are written grouped by layer, in increasing layer order, and
      // keep their relative order within a layer (via the per-layer cursors).
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
        auto entryInSoA = cellsView[layerCursor[layer]++];
        if (detector_ == "BH") {
          entryInSoA.dim1() = position.eta();
          float phi = position.phi();
          if (phi < 0.f) {
            phi += 2.f * static_cast<float>(M_PI);
          }
          entryInSoA.dim2() = phi;
        }  // else, isSilicon == true and eta phi values will not be used
        else {
          entryInSoA.dim1() = position.x();
          entryInSoA.dim2() = position.y();
        }
        entryInSoA.dim3() = position.z();
        entryInSoA.energy() = hgrh.energy();
        entryInSoA.mipEnergy() = hgrh.energy();  // TODO: CHANGE TO MIP
        entryInSoA.sigmaNoise() = sigmaNoise;
        entryInSoA.layer() = layer;
        entryInSoA.recHitIndex() = i;
        entryInSoA.detid() = detid.rawId();
        entryInSoA.time() = hgrh.time();
        entryInSoA.timeError() = hgrh.timeError();
      }
#if 0
        std::cout << "Size: " << cells->metadata().size() << " count cells: " << index
          << " i.e. " << cells->metadata().size() << std::endl;
#endif

      if constexpr (!std::is_same_v<Device, alpaka_common::DevHost>) {
        // Trigger copy async to GPU
        //std::cout << "GPU" << std::endl;
        HGCalSoARecHitsDeviceCollection deviceProduct{iEvent.queue(), cells->metadata().size()};
        alpaka::memcpy(iEvent.queue(), deviceProduct.buffer(), cells.const_buffer());
        iEvent.emplace(deviceToken_, std::move(deviceProduct));
      } else {
        //std::cout << "CPU" << std::endl;
        iEvent.emplace(deviceToken_, std::move(cells));
      }

      // Per-layer batch sizes for the downstream device clustering.
      iEvent.emplace(layerSizesToken_, std::move(layerSizes));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<std::string>("detector", "EE")->setComment("options EE, FH, BH,  HFNose; other value defaults to EE");
      desc.add<edm::InputTag>("recHits", edm::InputTag("HGCalRecHit", "HGCEERecHits"));
      desc.add<unsigned int>("maxNumberOfThickIndices", 6);
      desc.add<float>("fcPerEle", 0.00016020506);
      desc.add<std::vector<float>>("fcPerMip");
      desc.add<std::vector<float>>("thicknessCorrection");
      desc.add<std::vector<float>>("noises");
      desc.add<std::vector<float>>("dEdXweights");
      desc.add<double>("noiseMip", 0.2);
      desc.add<double>("sciThicknessCorrection", 1.0);
      desc.add<float>("ecut", 3.);
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    std::string detector_;
    bool initialized_;
    bool isNose_;
    unsigned maxNumberOfThickIndices_;
    unsigned int maxlayer_;
    int deltasi_index_regemfac_;
    float fcPerEle_;
    float ecut_;
    std::vector<float> fcPerMip_;
    std::vector<float> nonAgedNoises_;
    std::vector<float> dEdXweights_;
    std::vector<float> thicknessCorrection_;
    double noiseMip_;
    double sciThicknessCorrection_;
    std::vector<std::vector<double>> thresholds_;
    std::vector<std::vector<double>> v_sigmaNoise_;

    ticlgeom::Tools rhtools_;
    edm::ESGetToken<TICLGeomHost, CaloGeometryRecord> ticlGeomToken_;
    edm::ESGetToken<TICLGeomLookupHost, CaloGeometryRecord> ticlGeomLookupToken_;
    edm::ESGetToken<TICLGeomLayersHost, CaloGeometryRecord> ticlGeomLayersToken_;
    edm::EDGetTokenT<HGCRecHitCollection> hits_token_;
    device::EDPutToken<HGCalSoARecHitsDeviceCollection> const deviceToken_;
    edm::EDPutTokenT<std::vector<uint32_t>> const layerSizesToken_;

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
        if (!isNose_) {
          float scintillators_sigmaNoise = 0.001f * noiseMip_ * dEdXweights_[ilayer] / sciThicknessCorrection_;
          thresholds_[ilayer - 1][maxNumberOfThickIndices_] = ecut_ * scintillators_sigmaNoise;
          v_sigmaNoise_[ilayer - 1][maxNumberOfThickIndices_] = scintillators_sigmaNoise;
        }
      }
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(HGCalSoARecHitsProducer);

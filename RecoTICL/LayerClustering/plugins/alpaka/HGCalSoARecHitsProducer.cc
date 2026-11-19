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
          initialized_(false),
          isNose_(config.getParameter<std::string>("detector") == "HFNose"),
          isScintillator_(config.getParameter<std::string>("detector") == "BH"),
          maxNumberOfThickIndices_(config.getParameter<unsigned>("maxNumberOfThickIndices")),
          // Offset to jump from the CE-E silicon thickness indices to the CE-H ones in the thresholds array. It equals
          // the number of CE-E silicon thickness categories (3 for the pre-v19 geometries, 4 for v19).
          deltasi_index_regemfac_(maxNumberOfThickIndices_ / 2),
          fcPerEle_(config.getParameter<float>("fcPerEle")),
          ecut_(config.getParameter<float>("ecut")),
          fcPerMip_(config.getParameter<std::vector<float>>("fcPerMip")),
          nonAgedNoises_(config.getParameter<std::vector<float>>("noises")),
          dEdXweights_(config.getParameter<std::vector<float>>("dEdXweights")),
          thicknessCorrection_(config.getParameter<std::vector<float>>("thicknessCorrection")),
          noiseMip_(config.getParameter<edm::ParameterSet>("noiseMip").getParameter<double>("noise_MIP")),
          sciThicknessCorrection_(config.getParameter<double>("sciThicknessCorrection")),
          ticlGeomToken_(consumesCollector().esConsumes<TICLGeomHost, CaloGeometryRecord>(edm::ESInputTag("", ""))),
          hits_token_(consumes<HGCRecHitCollection>(config.getParameter<edm::InputTag>("recHits"))),
          deviceToken_{produces()},
          layerSizesToken_{produces("layerSizes")} {}

    ~HGCalSoARecHitsProducer() override = default;

    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override {
      edm::Handle<HGCRecHitCollection> hits_h;

      rhtools_.setGeometry(iSetup.getData(ticlGeomToken_));
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
      for (const HGCRecHit& hgrh : hits) {
        const RecHitPlacement placement = place(hgrh);
        if (placement.aboveThreshold) {
          hitsPerLayer[placement.layer]++;
          index++;
        }
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
        const RecHitPlacement placement = place(hgrh);
        if (not placement.aboveThreshold) {
          continue;
        }

        const DetId detid = hgrh.detid();
        const GlobalPoint position(rhtools_.getPosition(detid));
        auto entryInSoA = cellsView[layerCursor[placement.layer]++];
        // the scintillator rechits cluster in (eta, phi), with phi in [0, 2 pi), the silicon ones in (x, y)
        if (isScintillator_) {
          entryInSoA.dim1() = position.eta();
          float phi = position.phi();
          if (phi < 0.f) {
            phi += 2.f * static_cast<float>(M_PI);
          }
          entryInSoA.dim2() = phi;
        } else {
          entryInSoA.dim1() = position.x();
          entryInSoA.dim2() = position.y();
        }
        entryInSoA.dim3() = position.z();
        entryInSoA.energy() = hgrh.energy();
        // the energy in MIP is not computed here: the column holds the energy in GeV
        entryInSoA.mipEnergy() = hgrh.energy();
        entryInSoA.sigmaNoise() = v_sigmaNoise_.at(placement.layerOnSide).at(placement.thicknessIndex);
        entryInSoA.layer() = placement.layer;
        entryInSoA.recHitIndex() = i;
        entryInSoA.detid() = detid.rawId();
        entryInSoA.time() = hgrh.time();
        entryInSoA.timeError() = hgrh.timeError();
      }

      if constexpr (!std::is_same_v<Device, alpaka_common::DevHost>) {
        HGCalSoARecHitsDeviceCollection deviceProduct{iEvent.queue(), cells->metadata().size()};
        alpaka::memcpy(iEvent.queue(), deviceProduct.buffer(), cells.const_buffer());
        iEvent.emplace(deviceToken_, std::move(deviceProduct));
      } else {
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
      // the scintillator noise, as the noise PSet of the HGCAL scintillator digitizer: only noise_MIP is read
      edm::ParameterSetDescription descNoiseMip;
      descNoiseMip.add<bool>("scaleByDose", false);
      descNoiseMip.add<unsigned int>("scaleByDoseAlgo", 0);
      descNoiseMip.add<double>("scaleByDoseFactor", 1.);
      descNoiseMip.add<std::string>("doseMap", "");
      descNoiseMip.add<std::string>("sipmMap", "");
      descNoiseMip.add<double>("referenceIdark", -1.);
      descNoiseMip.add<double>("referenceXtalk", -1.);
      descNoiseMip.add<double>("noise_MIP", 1. / 100.);
      desc.add<edm::ParameterSetDescription>("noiseMip", descNoiseMip);
      desc.add<double>("sciThicknessCorrection", 1.0);
      desc.add<float>("ecut", 3.);
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    bool initialized_;
    const bool isNose_;
    const bool isScintillator_;
    unsigned maxNumberOfThickIndices_;
    unsigned int maxlayer_;
    int const deltasi_index_regemfac_;
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
    edm::EDGetTokenT<HGCRecHitCollection> hits_token_;
    device::EDPutToken<HGCalSoARecHitsDeviceCollection> const deviceToken_;
    edm::EDPutTokenT<std::vector<uint32_t>> const layerSizesToken_;

    // Where a rechit goes in the SoA and whether it is kept: its layer on its endcap, its thickness index (the
    // scintillator has none and gets the last one), its global layer index (layerOnSide + zside * maxlayer_), and
    // whether its energy passes the threshold, ecut times the noise of its sensor. Used by both loops of produce(),
    // which must agree: the first one counts the rechits of each layer, the second writes them.
    struct RecHitPlacement {
      unsigned int layerOnSide;
      int thicknessIndex;
      int layer;
      bool aboveThreshold;
    };

    RecHitPlacement place(const HGCRecHit& hgrh) const {
      const DetId detid = hgrh.detid();
      RecHitPlacement placement;
      placement.layerOnSide = rhtools_.getLayerWithOffset(detid) - 1;
      placement.thicknessIndex = rhtools_.getSiThickIndex(detid);
      if (placement.thicknessIndex == -1) {
        placement.thicknessIndex = maxNumberOfThickIndices_;
      }
      // the CE-H silicon thresholds come after the CE-E ones
      const bool isCEHSilicon = detid.det() == DetId::HGCalHSi || detid.subdetId() == HGCHEF;
      const int thresholdIndex = placement.thicknessIndex + (isCEHSilicon ? deltasi_index_regemfac_ : 0);
      placement.aboveThreshold = not(hgrh.energy() < thresholds_.at(placement.layerOnSide).at(thresholdIndex));
      placement.layer = placement.layerOnSide + ((rhtools_.zside(detid) + 1) >> 1) * maxlayer_;
      return placement;
    }

    void computeThreshold() {
      // Per layer, the thresholds and the noises: first the CE-E silicon thickness categories, then the CE-H ones,
      // then one slot for the scintillator.

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

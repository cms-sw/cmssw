// Implemented starting from TestAlpakaESProducerA.cc
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibTracker/Records/interface/SiStripDependentRecords.h"

#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"

#include "DataFormats/SiStripCommon/interface/ConstantsForHardwareSystems.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterTools.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizerConditionsSoA.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizerConditionsRecord.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/alpaka/SiStripClusterizerConditionsDevice.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip {
  // Unqualified lookup of the top level namespace
  using namespace ::sistrip;

  class DetToFed {
  public:
    DetToFed(uint32_t detid, uint16_t ipair, uint16_t fedid, uint8_t fedch)
        : detid_(detid), ipair_(ipair), fedid_(fedid), fedch_(fedch) {}
    inline uint32_t detID() const { return detid_; }
    inline uint16_t pair() const { return ipair_; }
    inline uint16_t fedID() const { return fedid_; }
    inline uint8_t fedCh() const { return fedch_; }

  private:
    uint32_t detid_;
    uint16_t ipair_;
    uint16_t fedid_;
    uint8_t fedch_;
  };

  class SiStripClusterizerConditionsESProducerAlpaka : public ESProducer {
  public:
    SiStripClusterizerConditionsESProducerAlpaka(edm::ParameterSet const& iConfig) : ESProducer(iConfig) {
      auto cc = setWhatProduced(this);
      gainsToken_ = cc.consumes();
      noisesToken_ = cc.consumes();
      qualityToken_ = cc.consumes(edm::ESInputTag{"", iConfig.getParameter<std::string>("QualityLabel")});
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<std::string>("QualityLabel", "");
      desc.add<std::string>("Label", "");
      descriptions.addWithDefaultLabel(desc);
    }

    std::unique_ptr<SiStripClusterizerConditionsHost> produce(SiStripClusterizerConditionsRecord const& iRecord) {
      auto gains = iRecord.getTransientHandle(gainsToken_);
      const auto& noises = iRecord.get(noisesToken_);
      const auto& quality = iRecord.get(qualityToken_);

      // Prepare the conditions on the host
      // Note: most likely depending on the size of this operation, perfomance could be much likely improved moving this operation in a device kernel.
      //       Some thinking should go into understanding how to optimize the move of gain/noise/quality and how to define a alpaka-kernel doing the
      //       preparation for the conditions.
      //       (same algo from RecoLocalTracker/SiStripClusterizer/plugins/SiStripClusterizerConditionsESProducer.cc)

      std::vector<DetToFed> detToFeds;  // detid_, ipair_, fedid_, fedch_

      std::vector<float> invthick(sistrip::NUMBER_OF_FEDS * sistrip::FEDCH_PER_FED);
      std::vector<uint32_t> detID(sistrip::NUMBER_OF_FEDS * sistrip::FEDCH_PER_FED);
      std::vector<uint16_t> iPair(sistrip::NUMBER_OF_FEDS * sistrip::FEDCH_PER_FED);
      std::vector<uint16_t> noise(sistrip::NUMBER_OF_FEDS * sistrip::FEDCH_PER_FED * sistrip::STRIPS_PER_FEDCH);
      std::vector<float> gain(sistrip::NUMBER_OF_FEDS * sistrip::APVS_PER_FEDCH * sistrip::FEDCH_PER_FED);

      fillSiStripClusterizerConditions_(
          quality, gains.product(), noises, detToFeds, invthick, detID, iPair, noise, gain);

      LogDebug("StripCondsESProd") << "Produced a SiStripClusterizerConditions object for " << detToFeds.size()
                                   << " modules";

      // Prepare the product & Fill the product
      const int DetToFeds_size = detToFeds.size();
      const int Data_fedch_size = detID.size();
      const int Data_strip_size = noise.size();
      const int Data_apv_size = gain.size();
      //// Typical sizes for these collections (from MC run)
      //// constexpr const unsigned int DetToFeds_size= 34813;
      //// constexpr const unsigned int Data_fedch_size= 42240;
      //// constexpr const unsigned int Data_strip_size= 10813440;
      //// constexpr const unsigned int Data_apv_size= 84480;

      assert(detID.size() == iPair.size() && iPair.size() == invthick.size());

      // PortableHostMultiCollection(const std::array<int32_t, members_>& sizes, alpaka_common::DevHost const& host)
      auto product = std::make_unique<SiStripClusterizerConditionsHost>(
          std::array<int32_t, 4>{{DetToFeds_size, Data_fedch_size, Data_strip_size, Data_apv_size}},
          cms::alpakatools::host());

      auto DetToFeds_View = product->view();
      auto Data_fedch_View = product->view<SiStripClusterizerConditionsData_fedchSoA>();
      auto Data_strip_View = product->view<SiStripClusterizerConditionsData_stripSoA>();
      auto Data_apv_View = product->view<SiStripClusterizerConditionsData_apvSoA>();

      for (int j = 0; j < DetToFeds_size; ++j) {
        auto entry = detToFeds[j];
        DetToFeds_View.detid_(j) = entry.detID();
        DetToFeds_View.fedch_(j) = entry.fedCh();
        DetToFeds_View.fedid_(j) = entry.fedID();
        DetToFeds_View.ipair_(j) = entry.pair();
      }
      for (int j = 0; j < Data_fedch_size; ++j) {
        Data_fedch_View.detID_(j) = detID[j];
        Data_fedch_View.iPair_(j) = iPair[j];
        Data_fedch_View.invthick_(j) = invthick[j];
      }
      for (int j = 0; j < Data_strip_size; ++j) {
        Data_strip_View.noise_(j) = noise[j];
      }
      for (int j = 0; j < Data_apv_size; ++j) {
        Data_apv_View.gain_(j) = gain[j];
      }

      return product;
    }

    // Auxiliary functions to translate indexes on the arrays
    static constexpr uint16_t badBit = 1 << 15;
    inline uint16_t fedIndex(uint16_t fed) { return (fed - FED_ID_MIN); };
    inline uint32_t stripIndex(uint16_t fed, uint8_t channel, uint16_t strip) {
      return (fedIndex(fed) * FEDCH_PER_FED * STRIPS_PER_FEDCH + channel * STRIPS_PER_FEDCH +
              (strip % STRIPS_PER_FEDCH));
    };
    inline uint32_t apvIndex(uint16_t fed, uint8_t channel, uint16_t strip) {
      return (fedIndex(fed) * APVS_PER_FEDCH * FEDCH_PER_FED + APVS_PER_CHAN * channel +
              (strip % STRIPS_PER_FEDCH) / STRIPS_PER_APV);
    };
    inline uint32_t channelIndex(uint16_t fed, uint8_t channel) { return (fedIndex(fed) * FEDCH_PER_FED + channel); };

  private:
    edm::ESGetToken<SiStripGain, SiStripGainRcd> gainsToken_;
    edm::ESGetToken<SiStripNoises, SiStripNoisesRcd> noisesToken_;
    edm::ESGetToken<SiStripQuality, SiStripQualityRcd> qualityToken_;

    // Make conditions as in the RecoLocalTracker/SiStripClusterizer/plugins/ClustersFromRawProducer.cc module
    void fillSiStripClusterizerConditions_(const SiStripQuality& quality,
                                           const SiStripGain* gains,
                                           const SiStripNoises& noises,
                                           std::vector<DetToFed>& detToFeds,
                                           std::vector<float>& invthick,
                                           std::vector<uint32_t>& detID,
                                           std::vector<uint16_t>& iPair,
                                           std::vector<uint16_t>& noise,
                                           std::vector<float>& gain);
  };

  void SiStripClusterizerConditionsESProducerAlpaka::fillSiStripClusterizerConditions_(const SiStripQuality& quality,
                                                                                       const SiStripGain* gains,
                                                                                       const SiStripNoises& noises,
                                                                                       std::vector<DetToFed>& detToFeds,
                                                                                       std::vector<float>& invthick,
                                                                                       std::vector<uint32_t>& detID,
                                                                                       std::vector<uint16_t>& iPair,
                                                                                       std::vector<uint16_t>& noise,
                                                                                       std::vector<float>& gain) {
    // Lambda to lookup channel index and set the invThickness array
    auto setInvThickness = [&](uint16_t fed, uint8_t channel, float val) {
      invthick[channelIndex(fed, channel)] = val;
    };

    // connected: map<DetID, std::vector<int>>
    // map of KEY=detid DATA=vector of apvs, maximum 6 APVs per detector module :
    const auto& connected = quality.cabling()->connected();
    // detCabling: map<DetID, std::vector<const FedChannelConnection*>
    // map of KEY=detid DATA=vector<FedChannelConnection>
    const auto& detCabling = quality.cabling()->getDetCabling();

    for (const auto& conn : connected) {
      const auto det = conn.first;
      if (!quality.IsModuleBad(det)) {
        const auto detConn_it = detCabling.find(det);

        if (detCabling.end() != detConn_it) {
          for (const auto& chan : (*detConn_it).second) {
            if (chan && chan->fedId() && chan->isConnected()) {
              const auto chan_detID = chan->detId();
              const auto chan_fedID = chan->fedId();
              const auto chan_fedCh = chan->fedCh();
              const auto chan_apvPairNumber = chan->apvPairNumber();

              detToFeds.emplace_back(chan_detID, chan_apvPairNumber, chan_fedID, chan_fedCh);
              // Fill the data structures
              // Note: the channelIndex is used to access the data structures
              detID[channelIndex(chan_fedID, chan_fedCh)] = chan_detID;
              iPair[channelIndex(chan_fedID, chan_fedCh)] = chan_apvPairNumber;
              setInvThickness(chan_fedID, chan_fedCh, siStripClusterTools::sensorThicknessInverse(chan_detID));

              auto offset = STRIPS_PER_FEDCH * chan_apvPairNumber;

              for (auto strip = 0; strip < STRIPS_PER_FEDCH; ++strip) {
                const auto gainRange = gains->getRange(det);

                const auto detstrip = strip + offset;
                const uint16_t strip_noise = SiStripNoises::getRawNoise(detstrip, noises.getRange(det));
                const auto strip_gain = SiStripGain::getStripGain(detstrip, gainRange);
                const auto bad = quality.IsStripBad(quality.getRange(det), detstrip);

                // gain is actually stored per-APV, not per-strip
                // setStrip_(chan_fedID, chan_fedCh, detstrip, noise, gain, bad);
                gain[apvIndex(chan_fedID, chan_fedCh, detstrip)] = strip_gain;
                if (bad) [[unlikely]] {
                  noise[stripIndex(chan_fedID, chan_fedCh, detstrip)] |= badBit;
                } else {
                  noise[stripIndex(chan_fedID, chan_fedCh, detstrip)] = strip_noise;
                }
              }
            }
          }
        }
      }
    }

    std::sort(detToFeds.begin(), detToFeds.end(), [](const DetToFed& a, const DetToFed& b) {
      return a.detID() < b.detID() || (a.detID() == b.detID() && a.pair() < b.pair());
    });
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(sistrip::SiStripClusterizerConditionsESProducerAlpaka);

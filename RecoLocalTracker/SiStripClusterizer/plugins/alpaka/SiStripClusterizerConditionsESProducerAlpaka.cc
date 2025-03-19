// Implemented starting from TestAlpakaESProducerA.cc

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestRecords.h"
#include "HeterogeneousCore/AlpakaTest/interface/ESTestData.h"
#include "HeterogeneousCore/AlpakaTest/interface/alpaka/AlpakaESTestData.h"

#include "RecoLocalTracker/Records/interface/SiStripClusterizerConditionsRcd.h"

#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/SiStripObjects/interface/SiStripClusterizerConditionsHost.h"
#include "CondFormats/SiStripObjects/interface/SiStripClusterizerConditionsSoA.h"

// #include "DataFormats/SiStripCommon/interface/Constants.h"
#include "DataFormats/SiStripCommon/interface/ConstantsForHardwareSystems.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterTools.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
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

  //  * This class demonstrates an ESProducer that uses the
  //  * PortableCollection-based data model, and that consumes a standard
  //  * host ESProduct and converts the data into PortableCollection, and
  //  * implicitly transfers the data product to device
  class SiStripClusterizerConditionsESProducerAlpaka : public ESProducer {
  public:
    SiStripClusterizerConditionsESProducerAlpaka(edm::ParameterSet const& iConfig) : ESProducer(iConfig) {
      invthick_.resize(sistrip::NUMBER_OF_FEDS * sistrip::FEDCH_PER_FED);
      detID_.resize(sistrip::NUMBER_OF_FEDS * sistrip::FEDCH_PER_FED);
      iPair_.resize(sistrip::NUMBER_OF_FEDS * sistrip::FEDCH_PER_FED);
      noise_.resize(sistrip::NUMBER_OF_FEDS * sistrip::FEDCH_PER_FED * sistrip::STRIPS_PER_FEDCH);
      gain_.resize(sistrip::NUMBER_OF_FEDS * sistrip::APVS_PER_FEDCH * sistrip::FEDCH_PER_FED);

      auto cc = setWhatProduced(this);
      gainsToken_ = cc.consumesFrom<SiStripGain, SiStripGainRcd>();
      noisesToken_ = cc.consumesFrom<SiStripNoises, SiStripNoisesRcd>();
      qualityToken_ = cc.consumesFrom<SiStripQuality, SiStripQualityRcd>(
          edm::ESInputTag{"", iConfig.getParameter<std::string>("QualityLabel")});
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<std::string>("QualityLabel", "");
      desc.add<std::string>("Label", "");
      descriptions.addWithDefaultLabel(desc);
    }

    std::optional<SiStripClusterizerConditionsHost> produce(SiStripClusterizerConditionsRcd const& iRecord) {
      auto gains = iRecord.getTransientHandle(gainsToken_);
      const auto& noises = iRecord.get(noisesToken_);
      const auto& quality = iRecord.get(qualityToken_);

      // const auto& connected = quality.cabling()->connected();
      // const auto& detCabling = quality.cabling()->getDetCabling();

      // Prepare the conditions on the host
      // Note: most likely depending on the size of this operation, perfomance could be much likely improved moving this operation in a device kernel.
      //       Some thinking should go into understanding how to optimize the move of gain/noise/quality and how to define a alpaka-kernel doing the
      //       preparation for the conditions.
      //       (same algo from RecoLocalTracker/SiStripClusterizer/plugins/SiStripClusterizerConditionsESProducer.cc)
      SiStripClusterizerConditionsOnHost(quality, gains.product(), noises);
      LogDebug("SiStripClusterizerConditionsESProducer")
          << "Produced a SiStripClusterizerConditions object for " << detToFeds_.size() << " modules";

      // Prepare the product & Fill the product
      const int DetToFedsSoA_size = detToFeds_.size();
      const int Data_fedch_SoA_size = detID_.size();
      const int Data_strip_SoA_size = noise_.size();
      const int Data_apv_SoA_size = gain_.size();

      assert(detID_.size() == iPair_.size() && iPair_.size() == invthick_.size());

      SiStripClusterizerConditionsHost product(
          {{DetToFedsSoA_size, Data_fedch_SoA_size, Data_strip_SoA_size, Data_apv_SoA_size}}, cms::alpakatools::host());

      auto DetToFeds_view = product.view();
      auto Data_fedchSoA_view = product.view<SiStripClusterizerConditionsData_fedchSoA>();
      auto Data_stripSoA_view = product.view<SiStripClusterizerConditionsData_stripSoA>();
      auto Data_apvSoA_view = product.view<SiStripClusterizerConditionsData_apvSoA>();

      for (int j = 0; j < DetToFedsSoA_size; ++j) {
        auto entry = detToFeds_[j];
        DetToFeds_view[j].detid_() = entry.detID();
        DetToFeds_view[j].fedch_() = entry.fedCh();
        DetToFeds_view[j].fedid_() = entry.fedID();
        DetToFeds_view[j].ipair_() = entry.pair();
      }
      for (int j = 0; j < Data_fedch_SoA_size; ++j) {
        Data_fedchSoA_view[j].detID_() = detID_[j];
        Data_fedchSoA_view[j].iPair_() = iPair_[j];
        Data_fedchSoA_view[j].invthick_() = invthick_[j];
      }
      for (int j = 0; j < Data_strip_SoA_size; ++j) {
        Data_stripSoA_view[j].noise_() = noise_[j];
      }
      for (int j = 0; j < Data_apv_SoA_size; ++j) {
        Data_apvSoA_view[j].gain_() = gain_[j];
      }
      return product;
    }

  private:
    edm::ESGetToken<SiStripGain, SiStripGainRcd> gainsToken_;
    edm::ESGetToken<SiStripNoises, SiStripNoisesRcd> noisesToken_;
    edm::ESGetToken<SiStripQuality, SiStripQualityRcd> qualityToken_;

    std::vector<DetToFed> detToFeds_;  // detis_, ipair_, fedid_, fedch_

    std::vector<uint32_t> detID_;
    std::vector<uint16_t> iPair_;
    std::vector<float> invthick_;

    std::vector<uint16_t> noise_;

    std::vector<float> gain_;

    void SiStripClusterizerConditionsOnHost(const SiStripQuality& quality,
                                            const SiStripGain* gains,
                                            const SiStripNoises& noises);

    inline uint16_t fedIndex(uint16_t fed) { return (fed - sistrip::FED_ID_MIN); };
    inline uint32_t stripIndex(uint16_t fed, uint8_t channel, uint16_t strip) {
      return (fedIndex(fed) * sistrip::FEDCH_PER_FED * sistrip::STRIPS_PER_FEDCH + channel * sistrip::STRIPS_PER_FEDCH +
              (strip % sistrip::STRIPS_PER_FEDCH));
    };
    inline uint32_t apvIndex(uint16_t fed, uint8_t channel, uint16_t strip) {
      return (fedIndex(fed) * sistrip::APVS_PER_FEDCH * sistrip::FEDCH_PER_FED + sistrip::APVS_PER_CHAN * channel +
              (strip % sistrip::STRIPS_PER_FEDCH) / sistrip::STRIPS_PER_APV);
    };
    inline uint32_t channelIndex(uint16_t fed, uint8_t channel) {
      return (fedIndex(fed) * sistrip::FEDCH_PER_FED + channel);
    };

    inline void setInvThickness(uint16_t fed, uint8_t channel, float invthick) {
      invthick_[channelIndex(fed, channel)] = invthick;
    };

    static constexpr uint16_t badBit = 1 << 15;

    void setStrip(uint16_t fed, uint8_t channel, uint16_t strip, uint16_t noise, float gain, bool bad);
  };

  void SiStripClusterizerConditionsESProducerAlpaka::SiStripClusterizerConditionsOnHost(const SiStripQuality& quality,
                                                                                        const SiStripGain* gains,
                                                                                        const SiStripNoises& noises) {
    // prepare conditions

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
              const auto detID = chan->detId();
              const auto fedID = chan->fedId();
              const auto fedCh = chan->fedCh();
              const auto iPair = chan->apvPairNumber();

              detToFeds_.emplace_back(detID, iPair, fedID, fedCh);

              detID_[channelIndex(fedID, fedCh)] = detID;
              iPair_[channelIndex(fedID, fedCh)] = iPair;
              setInvThickness(fedID, fedCh, siStripClusterTools::sensorThicknessInverse(detID));

              auto offset = 256 * iPair;

              for (auto strip = 0; strip < 256; ++strip) {
                const auto gainRange = gains->getRange(det);

                const auto detstrip = strip + offset;
                const uint16_t noise = SiStripNoises::getRawNoise(detstrip, noises.getRange(det));
                const auto gain = SiStripGain::getStripGain(detstrip, gainRange);
                const auto bad = quality.IsStripBad(quality.getRange(det), detstrip);

                // gain is actually stored per-APV, not per-strip
                setStrip(fedID, fedCh, detstrip, noise, gain, bad);
              }
            }
          }
        }
      }
    }
    std::sort(detToFeds_.begin(), detToFeds_.end(), [](const DetToFed& a, const DetToFed& b) {
      return a.detID() < b.detID() || (a.detID() == b.detID() && a.pair() < b.pair());
    });
  }

  void SiStripClusterizerConditionsESProducerAlpaka::setStrip(
      uint16_t fed, uint8_t channel, uint16_t strip, uint16_t noise, float gain, bool bad) {
    gain_[apvIndex(fed, channel, strip)] = gain;
    noise_[stripIndex(fed, channel, strip)] = noise;
    if (bad) {
      noise_[stripIndex(fed, channel, strip)] |= badBit;
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(SiStripClusterizerConditionsESProducerAlpaka);
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

#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizerConditionsRecord.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizerConditionsHostObject.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/alpaka/SiStripClusterizerConditionsDeviceObject.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip {
  // Unqualified lookup of the top level namespace
  using namespace ::sistrip;

  class DetToFed {
  public:
    DetToFed(uint32_t detid, uint16_t fedid, uint16_t fedch, uint16_t ipair)
        : detID_(detid), fedID_(fedid), fedCh_(fedch), iPair_(ipair) {}

    inline uint32_t detID() const { return detID_; }
    inline uint16_t fedID() const { return fedID_; }
    inline uint16_t fedCh() const { return fedCh_; }
    inline uint16_t iPair() const { return iPair_; }

  private:
    uint32_t detID_;
    uint16_t fedID_;
    uint16_t fedCh_;
    uint16_t iPair_;
  };

  class SiStripClusterizerConditionsESProducerAlpaka : public ESProducer {
  public:
    SiStripClusterizerConditionsESProducerAlpaka(edm::ParameterSet const& iConfig) : ESProducer(iConfig) {
      // Two tokens for quality because of the exc message:
      // You may need multiple tokens if you want to get the same data in multiple transitions.
      auto listenA = setWhatProduced(this, &SiStripClusterizerConditionsESProducerAlpaka::produceDetToFeds);
      qualityTokenA_ = listenA.consumes(iConfig.getParameter<edm::ESInputTag>("QualityLabel"));

      auto listenB = setWhatProduced(this, &SiStripClusterizerConditionsESProducerAlpaka::produceData);
      qualityTokenB_ = listenB.consumes(iConfig.getParameter<edm::ESInputTag>("QualityLabel"));
      gainsToken_ = listenB.consumes();
      noisesToken_ = listenB.consumes();
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::ESInputTag>("QualityLabel");
      desc.add<edm::ESInputTag>("Label");
      descriptions.addWithDefaultLabel(desc);
    }

    std::unique_ptr<SiStripClusterizerConditionsDetToFedsHostObject> produceDetToFeds(
        SiStripClusterizerConditionsDetToFedsRecord const& iRecord) {
      const auto& quality = iRecord.get(qualityTokenA_);

      auto product = std::make_unique<SiStripClusterizerConditionsDetToFedsHostObject>(cms::alpakatools::host());
      product->zeroInitialise();

      auto& detToFeds_qualityFlags = (*product).data()->qualityOk;

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
            for (const auto& conn : (*detConn_it).second) {
              if (conn && conn->fedId() && conn->isConnected()) {
                const auto conn_fedID = conn->fedId();
                const auto conn_fedCh = conn->fedCh();

                uint32_t idx = channelIndex(conn_fedID, conn_fedCh);
                detToFeds_qualityFlags[idx] = true;
                // ss << idx << ",";
              }
            }
          }
        }
      }

      return product;
    }

    std::unique_ptr<SiStripClusterizerConditionsGainNoiseCalsHostObject> produceData(
        SiStripClusterizerConditionsGainNoiseCalsRecord const& iRecord) {
      auto gains = iRecord.getTransientHandle(gainsToken_);
      auto noises = iRecord.getTransientHandle(noisesToken_);
      auto quality = iRecord.getTransientHandle(qualityTokenB_);

      // Prepare the conditions on the host
      auto product = std::make_unique<SiStripClusterizerConditionsGainNoiseCalsHostObject>(cms::alpakatools::host());
      // Fill the collections
      fillSiStripClusterizerConditions(quality.product(), gains.product(), noises.product(), *product->data());
      //
      return product;
    }

    // Auxiliary functions to translate indexes on the arrays
    static constexpr uint16_t badBit = 1 << 15;
    inline uint16_t fedIndex(uint16_t fed) { return (fed - FED_ID_MIN); };
    inline uint32_t stripIndex(uint16_t fed, uint16_t channel, uint16_t strip) {
      return (fedIndex(fed) * FEDCH_PER_FED * STRIPS_PER_FEDCH + channel * STRIPS_PER_FEDCH +
              (strip % STRIPS_PER_FEDCH));
    };
    inline uint32_t apvIndex(uint16_t fed, uint16_t channel, uint16_t strip) {
      return (fedIndex(fed) * APVS_PER_FEDCH * FEDCH_PER_FED + APVS_PER_CHAN * channel +
              (strip % STRIPS_PER_FEDCH) / STRIPS_PER_APV);
    };
    inline uint32_t channelIndex(uint16_t fedId, uint16_t fedCh) { return (fedIndex(fedId) * FEDCH_PER_FED + fedCh); };

  private:
    edm::ESGetToken<SiStripGain, SiStripGainRcd> gainsToken_;
    edm::ESGetToken<SiStripNoises, SiStripNoisesRcd> noisesToken_;
    // Two tokens for quality because of the exc message:
    // You may need multiple tokens if you want to get the same data in multiple transitions.
    edm::ESGetToken<SiStripQuality, SiStripQualityRcd> qualityTokenA_;
    edm::ESGetToken<SiStripQuality, SiStripQualityRcd> qualityTokenB_;

    // Make conditions as in the RecoLocalTracker/SiStripClusterizer/plugins/ClustersFromRawProducer.cc module
    void fillSiStripClusterizerConditions(const SiStripQuality* quality,
                                          const SiStripGain* gains,
                                          const SiStripNoises* noises,
                                          GainNoiseCals& calibs);
  };

  void SiStripClusterizerConditionsESProducerAlpaka::fillSiStripClusterizerConditions(const SiStripQuality* quality,
                                                                                      const SiStripGain* gains,
                                                                                      const SiStripNoises* noises,
                                                                                      GainNoiseCals& calibs) {
    // Alias the members
    auto& invthick = calibs.invthick;
    auto& detID = calibs.detID;
    auto& iPair = calibs.iPair;
    auto& noise = calibs.noise;
    auto& gain = calibs.gain;

    // connected: map<DetID, std::vector<int>>
    // map of KEY=detid DATA=vector of apvs, maximum 6 APVs per detector module :
    const auto& connected = quality->cabling()->connected();
    // detCabling: map<DetID, std::vector<const FedChannelConnection*>
    // map of KEY=detid DATA=vector<FedChannelConnection>
    const auto& detCabling = quality->cabling()->getDetCabling();

    for (const auto& conn : connected) {
      const auto det = conn.first;

      if (!quality->IsModuleBad(det)) {
        const auto detConn_it = detCabling.find(det);

        const auto gainRange = gains->getRange(det);
        if (detCabling.end() != detConn_it) {
          for (const auto& chan : (*detConn_it).second) {
            if (chan && chan->fedId() && chan->isConnected()) {
              const uint32_t chan_detID = chan->detId();
              const uint16_t chan_fedID = chan->fedId();
              const uint16_t chan_fedCh = chan->fedCh();
              const uint16_t chan_apvPairNumber = chan->apvPairNumber();

              // Fill the data structures
              // Note: the channelIndex is used to access the data structures
              detID[channelIndex(chan_fedID, chan_fedCh)] = chan_detID;
              iPair[channelIndex(chan_fedID, chan_fedCh)] = chan_apvPairNumber;
              invthick[channelIndex(chan_fedID, chan_fedCh)] = siStripClusterTools::sensorThicknessInverse(chan_detID);

              auto offset = STRIPS_PER_FEDCH * chan_apvPairNumber;
              for (uint16_t strip = 0; strip < STRIPS_PER_FEDCH; ++strip) {
                const auto detstrip = strip + offset;
                const uint16_t strip_noise = SiStripNoises::getRawNoise(detstrip, noises->getRange(det));
                const auto bad = quality->IsStripBad(quality->getRange(det), detstrip);

                // setStrip_(chan_fedID, chan_fedCh, detstrip, noise, gain, bad);
                if (bad) [[unlikely]] {
                  noise[stripIndex(chan_fedID, chan_fedCh, detstrip)] = badBit;
                } else {
                  noise[stripIndex(chan_fedID, chan_fedCh, detstrip)] = strip_noise;
                }
              }

              // gain is actually stored per-APV, not per-strip (so stored for the strpis 0-127 and 128-255)
              gain[apvIndex(chan_fedID, chan_fedCh, 0)] = SiStripGain::getApvGain(0, gainRange);
              gain[apvIndex(chan_fedID, chan_fedCh, 255)] = SiStripGain::getApvGain(1, gainRange);
            }
          }
        }
      }
    }
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(sistrip::SiStripClusterizerConditionsESProducerAlpaka);

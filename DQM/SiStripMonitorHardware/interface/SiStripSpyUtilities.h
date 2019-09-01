#ifndef DQM_SiStripMonitorHardware_SiStripSpyUtilities_H
#define DQM_SiStripMonitorHardware_SiStripSpyUtilities_H

// Standard includes.
//#include <utility>
#include <string>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"

#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"

// Other classes
class EventSetup;

namespace sistrip {

  class SpyUtilities {
  public:
    struct Frame {
      uint32_t detId;
      uint16_t digitalLow;
      uint16_t digitalHigh;
      uint16_t firstHeaderBit;
      uint16_t firstTrailerBit;
      float baseline;
      std::pair<bool, bool> apvErrorBit;
      std::pair<uint8_t, uint8_t> apvAddress;
    };

    struct FrameQuality {
      uint16_t minDigiRange;
      uint16_t maxDigiRange;
      uint16_t minZeroLight;
      uint16_t maxZeroLight;
      uint16_t minTickHeight;
      uint16_t maxTickHeight;
    };

    SpyUtilities();
    ~SpyUtilities();

    //get cabling for an eventSetup: internal counter to see if cabling has changed.
    const SiStripFedCabling* getCabling(const edm::EventSetup&);     //!< Updates the cabling object from the DB.
    const SiStripDetCabling* getDetCabling(const edm::EventSetup&);  //!< Updates the det cabling object from the DB.

    edm::ESHandle<SiStripPedestals> getPedestalHandle(const edm::EventSetup& eventSetup);
    edm::ESHandle<SiStripNoises> getNoiseHandle(const edm::EventSetup& eventSetup);

    //fill variables from frame
    static const Frame extractFrameInfo(const edm::DetSetVector<SiStripRawDigi>::detset& channelDigis,
                                        bool aPrintDebug = false);

    static void getMajorityHeader(const edm::DetSetVector<SiStripRawDigi>* aInputDigis,
                                  uint16_t& firstHeaderBit,
                                  bool printResult = true);

    //check frame is valid
    static const bool isValid(const Frame& aFrame, const FrameQuality& aQuality, const uint16_t aExpectedPos);

    //extract range, threshold and apvAddress
    static const uint16_t range(const Frame& aFrame);

    static const uint16_t threshold(const Frame& aFrame);

    static const uint8_t extractAPVaddress(const Frame& aFrame);

    //find position of the first header and trailer bit
    static const uint16_t findHeaderBits(const edm::DetSetVector<SiStripRawDigi>::detset& channelDigis,
                                         const uint16_t threshold);

    static const uint16_t findTrailerBits(const edm::DetSetVector<SiStripRawDigi>::detset& channelDigis,
                                          const uint16_t threshold);

    //find both APV addresses and error bits
    static const std::pair<bool, bool> findAPVErrorBits(const edm::DetSetVector<SiStripRawDigi>::detset& channelDigis,
                                                        const uint16_t threshold,
                                                        const uint16_t aFirstBits);

    static const std::pair<uint8_t, uint8_t> findAPVAddresses(
        const edm::DetSetVector<SiStripRawDigi>::detset& channelDigis,
        const uint16_t threshold,
        const uint16_t aFirstBits);

    static std::string print(const Frame& aFrame, std::string aErr);

    static void fedIndex(uint32_t aFedIndex, uint16_t& aFedId, uint16_t& aFedChannel);

    static std::pair<uint16_t, uint32_t> findMajorityValue(std::vector<uint16_t>& values, const uint16_t aFedId = 0);

    static void fillFEDMajorities(const std::map<uint32_t, uint32_t>& channelValues,
                                  std::vector<uint32_t>& fedMajoritiesToFill);

  private:
    // Cabling
    const SiStripFedCabling* cabling_;  //!< The cabling object.
    uint32_t cacheId_;                  //!< DB cache ID used to establish if the cabling has changed during the run.

    // DetCabling
    const SiStripDetCabling* detCabling_;  //!< The cabling object.
    uint32_t cacheIdDet_;                  //!< DB cache ID used to establish if the cabling has changed during the run.

    //used to see if the pedestals have changed.
    uint32_t pedsCacheId_;
    edm::ESHandle<SiStripPedestals> pedsHandle_;

    //used to see if the noises have changed.
    uint32_t noiseCacheId_;
    edm::ESHandle<SiStripNoises> noiseHandle_;
  };

}  // namespace sistrip

#endif  // DQM_SiStripMonitorHardware_SiStripSpyUtilities_H

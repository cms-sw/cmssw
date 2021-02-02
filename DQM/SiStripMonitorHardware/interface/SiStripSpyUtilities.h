#ifndef DQM_SiStripMonitorHardware_SiStripSpyUtilities_H
#define DQM_SiStripMonitorHardware_SiStripSpyUtilities_H

// Standard includes.
//#include <utility>
#include <string>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"

// Other classes
class EventSetup;

namespace sistrip {

  namespace SpyUtilities {
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

    //fill variables from frame
    const Frame extractFrameInfo(const edm::DetSetVector<SiStripRawDigi>::detset& channelDigis,
                                 bool aPrintDebug = false);

    void getMajorityHeader(const edm::DetSetVector<SiStripRawDigi>* aInputDigis,
                           uint16_t& firstHeaderBit,
                           bool printResult = true);

    //check frame is valid
    const bool isValid(const Frame& aFrame, const FrameQuality& aQuality, const uint16_t aExpectedPos);

    //extract range, threshold and apvAddress
    const uint16_t range(const Frame& aFrame);

    const uint16_t threshold(const Frame& aFrame);

    const uint8_t extractAPVaddress(const Frame& aFrame);

    //find position of the first header and trailer bit
    const uint16_t findHeaderBits(const edm::DetSetVector<SiStripRawDigi>::detset& channelDigis,
                                  const uint16_t threshold);

    const uint16_t findTrailerBits(const edm::DetSetVector<SiStripRawDigi>::detset& channelDigis,
                                   const uint16_t threshold);

    //find both APV addresses and error bits
    const std::pair<bool, bool> findAPVErrorBits(const edm::DetSetVector<SiStripRawDigi>::detset& channelDigis,
                                                 const uint16_t threshold,
                                                 const uint16_t aFirstBits);

    const std::pair<uint8_t, uint8_t> findAPVAddresses(const edm::DetSetVector<SiStripRawDigi>::detset& channelDigis,
                                                       const uint16_t threshold,
                                                       const uint16_t aFirstBits);

    std::string print(const Frame& aFrame, std::string aErr);

    void fedIndex(uint32_t aFedIndex, uint16_t& aFedId, uint16_t& aFedChannel);

    std::pair<uint16_t, uint32_t> findMajorityValue(std::vector<uint16_t>& values, const uint16_t aFedId = 0);

    void fillFEDMajorities(const std::map<uint32_t, uint32_t>& channelValues,
                           std::vector<uint32_t>& fedMajoritiesToFill);

  }  // namespace SpyUtilities
}  // namespace sistrip

#endif  // DQM_SiStripMonitorHardware_SiStripSpyUtilities_H

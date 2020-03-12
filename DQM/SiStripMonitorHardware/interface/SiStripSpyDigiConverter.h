#ifndef DQM_SiStripMonitorHardware_SiStripSpyDigiConverter_H
#define DQM_SiStripMonitorHardware_SiStripSpyDigiConverter_H

#include <memory>
#include <vector>

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"

#include "DQM/SiStripMonitorHardware/interface/SiStripSpyUtilities.h"
#include <cstdint>

// Forward define other classes
class SiStripFedCabling;

namespace sistrip {

  /*! \brief Converts scope mode like digis into virgin raw like digis by:
     *  -extracting the frame payload,
     *  -reordering to physical order and
     *  -merging the DetSets indexed by FedKey to DetSets indexed by DetId
     *
     *
     */
  class SpyDigiConverter {
  public:
    typedef edm::DetSetVector<SiStripRawDigi> DSVRawDigis;

    //all methods are static so no instances are needed but allow anyway
    SpyDigiConverter() {}
    ~SpyDigiConverter() {}

    /*! \brief Extract frames from the scope digis.
         *
         * If pAPVAddress is set, the map is filled with a map from FedKey to APVAddress.
         * minAllowedRange is the min allowed range of digis when determine the threshold.
         */
    static std::unique_ptr<DSVRawDigis> extractPayloadDigis(const DSVRawDigis* inputScopeDigis,
                                                            std::vector<uint32_t>* pAPVAddresses,
                                                            const bool discardDigisWithAPVAddrErr,
                                                            const sistrip::SpyUtilities::FrameQuality& aQuality,
                                                            const uint16_t expectedPos);

    /* \brief Reorder from readout order to physical order */
    static std::unique_ptr<DSVRawDigis> reorderDigis(const DSVRawDigis* inputPayloadDigis);

    /* \brief Merge channel digis into modules. */
    static std::unique_ptr<DSVRawDigis> mergeModuleChannels(const DSVRawDigis* inputPhysicalOrderChannelDigis,
                                                            const SiStripFedCabling& cabling);

  private:
    typedef DSVRawDigis::detset DetSetRawDigis;

    static void processFED(const uint16_t aPreviousFedId,
                           const bool discardDigisWithAPVAddrErr,
                           std::vector<uint32_t>* pAPVAddresses,
                           std::vector<DetSetRawDigis>& outputData,
                           std::vector<uint16_t>& aAddrVec,
                           std::vector<uint16_t>& aHeaderBitVec,
                           std::vector<uint16_t>& aTrailBitVec,
                           std::vector<DSVRawDigis::const_iterator>& aFedScopeDigis);

  };  // end of SpyDigiConverter class.

}  // namespace sistrip

#endif  // DQM_SiStripMonitorHardware_SiStripSpyDigiConverter_H

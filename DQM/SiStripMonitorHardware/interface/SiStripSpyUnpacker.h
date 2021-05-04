#ifndef DQM_SiStripMonitorHardware_SiStripSpyUnpacker_H
#define DQM_SiStripMonitorHardware_SiStripSpyUnpacker_H

#include "DataFormats/Common/interface/DetSetVector.h"

// Standard includes.
#include <vector>
#include <utility>
#include <cstdint>

// Other classes
class FEDRawDataCollection;
class FEDRawData;
class SiStripRawDigi;
class SiStripFedCabling;

namespace sistrip {

  /*! \brief Unpacks spy channel data into scope mode-like digis
     * @author Nick Cripps
     * @date Autumn 2009
     * 
     * TODO: Implement unpacking by detID.
     *
     * --- Modifications: 
     * ------ A.-M. Magnan, 11/03/10: change Counters map to vectors: 
     * ------                         save space and provide faster access to elements.
     * ------ A.-M. Magnan, 25/02/10: add run number to the event
     * ------ A.-M. Magnan, 13/01/10: change Counters map to be keyed by fedid
     */
  class SpyUnpacker {
  public:
    typedef edm::DetSetVector<SiStripRawDigi> RawDigis;
    typedef std::vector<uint32_t> Counters;

    SpyUnpacker(const bool allowIncompleteEvents);  //!< Constructor.
    ~SpyUnpacker();                                 //!< Destructor.

    /*! \brief Creates the scope mode digis for the supplied FED IDs or detIds and stores event counters.
         *
         * If FED IDs are supplied (useFedId=true), unpacks all channels found in the cabling with data.
         * If an empty vector of IDs is supplied, it unpacks all it can find in the FEDRawDataCollection.
         */
    void createDigis(const SiStripFedCabling&,
                     const FEDRawDataCollection&,
                     RawDigis* pDigis,
                     const std::vector<uint32_t>& ids,
                     Counters* pTotalEventCounts,
                     Counters* pL1ACounts,
                     uint32_t* aRunRef);

  private:
    // Configuration
    const bool allowIncompleteEvents_;

  };  // end of SpyUnpacker class.

}  // namespace sistrip

#endif  // DQM_SiStripMonitorHardware_SiStripSpyUnpacker_H

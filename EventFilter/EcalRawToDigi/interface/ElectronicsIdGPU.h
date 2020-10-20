#ifndef EventFilter_EcalRawToDigi_interface_ElectronicsIdGPU_h
#define EventFilter_EcalRawToDigi_interface_ElectronicsIdGPU_h

#include <cstdint>

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

namespace ecal {
  namespace raw {

    /** \brief Ecal readout channel identification
    [32:20] Unused (so far)
    [19:13]  DCC id
    [12:6]   tower
    [5:3]    strip
    [2:0]    xtal
    Index starts from 1
 */

    class ElectronicsIdGPU {
    public:
      /** Default constructor -- invalid value */
      constexpr ElectronicsIdGPU() : id_{0xFFFFFFFFu} {}
      /** from raw */
      constexpr ElectronicsIdGPU(uint32_t id) : id_{id} {}
      /** Constructor from dcc,tower,channel **/
      constexpr ElectronicsIdGPU(uint8_t const dccid, uint8_t const towerid, uint8_t const stripid, uint8_t const xtalid)
          : id_{static_cast<uint32_t>((xtalid & 0x7) | ((stripid & 0x7) << 3) | ((towerid & 0x7F) << 6) |
                                      ((dccid & 0x7F) << 13))} {}

      constexpr uint32_t operator()() { return id_; }
      constexpr uint32_t rawId() const { return id_; }

      /// get the DCC (Ecal Local DCC value not global one) id
      constexpr uint8_t dccId() const { return (id_ >> 13) & 0x7F; }
      /// get the tower id
      constexpr uint8_t towerId() const { return (id_ >> 6) & 0x7F; }
      /// get the tower id
      constexpr uint8_t stripId() const { return (id_ >> 3) & 0x7; }
      /// get the channel id
      constexpr uint8_t xtalId() const { return (id_ & 0x7); }

      /// get the subdet
      //EcalSubdetector subdet() const;

      /// get a fast, compact, unique index for linear lookups (maximum value = 4194303)
      constexpr uint32_t linearIndex() const { return id_ & 0x3FFFFF; }

      /// so far for EndCap only :
      //int channelId() const;  // xtal id between 1 and 25

      static constexpr int kTowersInPhi = 4;     // see EBDetId
      static constexpr int kCrystalsInPhi = 20;  // see EBDetId

      static constexpr uint8_t MAX_DCCID = 54;  //To be updated with correct and final number
      static constexpr uint8_t MIN_DCCID = 1;
      static constexpr uint8_t MAX_TOWERID = 70;
      static constexpr uint8_t MIN_TOWERID = 1;
      static constexpr uint8_t MAX_STRIPID = 5;
      static constexpr uint8_t MIN_STRIPID = 1;
      static constexpr uint8_t MAX_CHANNELID = 25;
      static constexpr uint8_t MIN_CHANNELID = 1;
      static constexpr uint8_t MAX_XTALID = 5;
      static constexpr uint8_t MIN_XTALID = 1;

      static constexpr int MIN_DCCID_EEM = 1;
      static constexpr int MAX_DCCID_EEM = 9;
      static constexpr int MIN_DCCID_EBM = 10;
      static constexpr int MAX_DCCID_EBM = 27;
      static constexpr int MIN_DCCID_EBP = 28;
      static constexpr int MAX_DCCID_EBP = 45;
      static constexpr int MIN_DCCID_EEP = 46;
      static constexpr int MAX_DCCID_EEP = 54;

      static constexpr int DCCID_PHI0_EBM = 10;
      static constexpr int DCCID_PHI0_EBP = 28;

      static constexpr int kDCCChannelBoundary = 17;
      static constexpr int DCC_EBM = 10;  // id of the DCC in EB- which contains phi=0 deg.
      static constexpr int DCC_EBP = 28;  // id of the DCC in EB+ which contains phi=0 deg.
      static constexpr int DCC_EEM = 1;   // id of the DCC in EE- which contains phi=0 deg.
      static constexpr int DCC_EEP = 46;  // id of the DCC in EE+ which contains phi=0 deg.

    private:
      uint32_t id_;
    };

  }  // namespace raw
}  // namespace ecal

#endif  // EventFilter_EcalRawToDigi_interface_ElectronicsIdGPU_h

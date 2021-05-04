
#ifndef CalibFormats_SiStripObjects_SiStripModule_H
#define CalibFormats_SiStripObjects_SiStripModule_H

#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include <map>
#include <ostream>
#include <sstream>
#include <vector>
#include <cstdint>

class SiStripModule;

/** Debug info for SiStripModule class. */
std::ostream &operator<<(std::ostream &, const SiStripModule &);

/**
    @class SiStripModule
    @author R.Bainbridge
    @brief Device and connection information at the level of a
    front-end module.
*/
class SiStripModule {
public:
  // ---------- Constructors and adding devices ----------

  /** Constructor. */
  SiStripModule(const FedChannelConnection &conn);

  /** Default constructor. */
  ~SiStripModule() { ; }

  /** Sets device info (addresses, DetID, etc) for this module. */
  void addDevices(const FedChannelConnection &conn);

  // ---------- Typedefs and enums ----------

  /** Pair containing FED id/channel. */
  typedef std::pair<uint16_t, uint16_t> PairOfU16;

  /** Struct containing FED crate/slot/id/channel. */
  // typedef PairOfU16 FedChannel;
  class FedChannel {
  public:
    uint16_t fedCrate_;
    uint16_t fedSlot_;
    uint16_t fedId_;
    uint16_t fedCh_;
    FedChannel(const uint16_t &crate, const uint16_t &slot, const uint16_t &id, const uint16_t &ch)
        : fedCrate_(crate), fedSlot_(slot), fedId_(id), fedCh_(ch) {
      ;
    }
    FedChannel() : fedCrate_(0), fedSlot_(0), fedId_(0), fedCh_(0) { ; }
  };

  /** Map between LLD channel and FED channel */
  typedef std::map<uint16_t, FedChannel> FedCabling;

  // ---------- Control structure ----------

  inline const uint16_t &fecCrate() const;
  inline const uint16_t &fecSlot() const;
  inline const uint16_t &fecRing() const;
  inline const uint16_t &ccuAddr() const;
  inline const uint16_t &ccuChan() const;

  /** Returns control "key" for this module, containing address
      information on FEC crate, slot, ring, CCU, and module. */
  inline const SiStripFecKey &key() const;

  // ---------- APV devices ----------

  /** Returns I2C addresses of active ("found") APVs. */
  std::vector<uint16_t> activeApvs() const;

  /** Identifies whether APV of a given I2C address (32->37) or
      footprint position on the hybrid (0->5) is active or
      not. Returns device I2C address or zero if not active. */
  const uint16_t &activeApv(const uint16_t &apv_address) const;

  /** Identifies APV pairs that are active for given LLD channel
      (1->3). Returns device I2C address or zero if not active. */
  PairOfU16 activeApvPair(const uint16_t &lld_channel) const;

  /** Add APV to module using I2C address (32->37). */
  void addApv(const uint16_t &apv_address);

  // ---------- Other hybrid devices ----------

  /** Identifies whether the DCU device is active ("found") or not. */
  inline const uint16_t &dcu() const;

  /** Identifies whether the MUX device is active ("found") or not. */
  inline const uint16_t &mux() const;

  /** Identifies whether the PLL device is active ("found") or not. */
  inline const uint16_t &pll() const;

  /** Identifies whether the LLD device is active ("found") or not. */
  inline const uint16_t &lld() const;

  // ---------- Module information ----------

  /** Returns DCU id for this module. */
  inline const uint32_t &dcuId() const;

  /** Returns LLD channel (1->3) for given APV pair (0->1 or 0->2). */
  uint16_t lldChannel(const uint16_t &apv_pair_num) const;

  /** Set DCU id for this module. */
  inline void dcuId(const uint32_t &dcu_id);

  // ---------- Detector information ----------

  /** Returns unique (geometry-based) identifier for this module. */
  inline const uint32_t &detId() const;

  /** Returns APV pair (0->1 or 0->2) for given LLD channel (1->3). */
  uint16_t apvPairNumber(const uint16_t &lld_channel) const;

  /** Returns number of APV pairs for this module. */
  inline const uint16_t &nApvPairs() const;

  /** Returns number of detector strips for this module. */
  inline uint16_t nDetStrips() const;

  /** Set DetId for this module. */
  inline void detId(const uint32_t &det_id);

  /** Set number of detector strips for this module. */
  void nApvPairs(const uint16_t &npairs);

  // ---------- FED connection information ----------

  /** Returns map of apvPairNumber and FedChannel. */
  inline const FedCabling &fedChannels() const;

  /** Returns FedChannel for a given apvPairNumber. */
  FedChannel fedCh(const uint16_t &apv_pair_num) const;

  /** Sets FedChannel for given APV address (32->37). Returns true
      if connection made, false otherwise. */
  bool fedCh(const uint16_t &apv_address, const FedChannel &fed_ch);

  // ---------- Miscellaneous ----------

  /** Prints some debug information for this module. */
  void print(std::stringstream &) const;

  /** Prints some terse debug information for this module. */
  void terse(std::stringstream &) const;

  /** Returns cable length. */
  inline const uint16_t &length() const;

  /** Sets cable length. */
  inline void length(const uint16_t &length);

private:
  /** Control key/path for this module. */
  SiStripFecKey key_;

  // APVs found (identified by decimal I2C address)
  uint16_t apv32_;
  uint16_t apv33_;
  uint16_t apv34_;
  uint16_t apv35_;
  uint16_t apv36_;
  uint16_t apv37_;

  // Devices found (with hex addr)
  uint16_t dcu0x00_;
  uint16_t mux0x43_;
  uint16_t pll0x44_;
  uint16_t lld0x60_;

  // Detector
  uint32_t dcuId_;
  uint32_t detId_;
  uint16_t nApvPairs_;

  /** KEY = LLD channel, DATA = FedId + FedCh */
  FedCabling cabling_;
  uint16_t length_;
};

// --------------- inline methods ---------------

const uint16_t &SiStripModule::fecCrate() const { return key_.fecCrate(); }
const uint16_t &SiStripModule::fecSlot() const { return key_.fecSlot(); }
const uint16_t &SiStripModule::fecRing() const { return key_.fecRing(); }
const uint16_t &SiStripModule::ccuAddr() const { return key_.ccuAddr(); }
const uint16_t &SiStripModule::ccuChan() const { return key_.ccuChan(); }

const SiStripFecKey &SiStripModule::key() const { return key_; }

const uint32_t &SiStripModule::dcuId() const { return dcuId_; }
const uint32_t &SiStripModule::detId() const { return detId_; }
const uint16_t &SiStripModule::nApvPairs() const { return nApvPairs_; }
uint16_t SiStripModule::nDetStrips() const { return 256 * nApvPairs_; }

void SiStripModule::dcuId(const uint32_t &dcu_id) {
  if (dcu_id) {
    dcuId_ = dcu_id;
    dcu0x00_ = true;
  }
}
void SiStripModule::detId(const uint32_t &det_id) {
  if (det_id) {
    detId_ = det_id;
  }
}
const SiStripModule::FedCabling &SiStripModule::fedChannels() const { return cabling_; }

const uint16_t &SiStripModule::length() const { return length_; }
void SiStripModule::length(const uint16_t &length) { length_ = length; }

const uint16_t &SiStripModule::dcu() const { return dcu0x00_; }
const uint16_t &SiStripModule::mux() const { return mux0x43_; }
const uint16_t &SiStripModule::pll() const { return pll0x44_; }
const uint16_t &SiStripModule::lld() const { return lld0x60_; }

#endif  // CalibTracker_SiStripObjects_SiStripModule_H

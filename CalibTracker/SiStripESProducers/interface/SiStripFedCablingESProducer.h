#ifndef CalibTracker_SiStripESProducers_SiStripFedCablingESProducer_H
#define CalibTracker_SiStripESProducers_SiStripFedCablingESProducer_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <memory>
#include <cstdint>

class SiStripFedCabling;
class SiStripFedCablingRcd;

/** 
    @class SiStripFedCablingESProducer
    @author R.Bainbridge
    @brief Abstract base class for producer of SiStripFedCabling record.
*/
class SiStripFedCablingESProducer : public edm::ESProducer {
public:
  SiStripFedCablingESProducer(const edm::ParameterSet&);
  ~SiStripFedCablingESProducer() override;

  /** Calls pure virtual make() method, to force concrete implementation. */
  virtual std::unique_ptr<SiStripFedCabling> produce(const SiStripFedCablingRcd&);

  SiStripFedCablingESProducer(const SiStripFedCablingESProducer&) = delete;
  const SiStripFedCablingESProducer& operator=(const SiStripFedCablingESProducer&) = delete;

private:
  virtual SiStripFedCabling* make(const SiStripFedCablingRcd&) = 0;

public:
  // Utility methods that generate "fake" control structure numbering
  static uint16_t fecCrate(const uint16_t& nth_module);  // 4 crates within system
  static uint16_t fecSlot(const uint16_t& nth_module);   // 11 FECs per crate
  static uint16_t fecRing(const uint16_t& nth_module);   // 8 control rings per FEC
  static uint16_t ccuAddr(const uint16_t& nth_module);   // 8 CCU modules per control ring
  static uint16_t ccuChan(const uint16_t& nth_module);   // 8 modules per CCU
};

// ---------- inline methods ----------

inline uint16_t SiStripFedCablingESProducer::fecCrate(const uint16_t& module) {
  return (module / (8 * 8 * 8 * 11)) % 4 + 1;
}
inline uint16_t SiStripFedCablingESProducer::fecSlot(const uint16_t& module) { return (module / (8 * 8 * 8)) % 11 + 2; }
inline uint16_t SiStripFedCablingESProducer::fecRing(const uint16_t& module) { return (module / (8 * 8)) % 8 + 1; }
inline uint16_t SiStripFedCablingESProducer::ccuAddr(const uint16_t& module) { return (module / 8) % 8 + 1; }
inline uint16_t SiStripFedCablingESProducer::ccuChan(const uint16_t& module) { return (module / 1) % 8 + 16; }

#endif  // CalibTracker_SiStripESProducers_SiStripFedCablingESProducer_H

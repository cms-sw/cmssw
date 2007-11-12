#ifndef CalibTracker_SiStripESProducers_SiStripFedCablingESProducer_H
#define CalibTracker_SiStripESProducers_SiStripFedCablingESProducer_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "boost/cstdint.hpp"
#include <memory>

class SiStripFedCabling;
class SiStripFedCablingRcd;

/** 
    @class SiStripFedCablingESProducer
    @brief Pure virtual class for EventSetup sources of SiStripFedCabling.
    @author R.Bainbridge
*/
class SiStripFedCablingESProducer : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {

 public:

  SiStripFedCablingESProducer( const edm::ParameterSet& );
  virtual ~SiStripFedCablingESProducer() {;}
  
  virtual std::auto_ptr<SiStripFedCabling> produce( const SiStripFedCablingRcd& );
  
  // Utility methods that generate control structure numbering
  static inline uint16_t fecCrate( const uint16_t& module ); // 4 crates within system
  static inline uint16_t fecSlot( const uint16_t& module );  // 11 FECs per crate
  static inline uint16_t fecRing( const uint16_t& module );  // 8 control rings per FEC
  static inline uint16_t ccuAddr( const uint16_t& module );  // 8 CCUMs per control ring
  static inline uint16_t ccuChan( const uint16_t& module );  // 8 modules per CCUM
  
 protected:
  
  virtual void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
			       const edm::IOVSyncValue&,
			       edm::ValidityInterval& );
  
 private:
  
  SiStripFedCablingESProducer( const SiStripFedCablingESProducer& );
  const SiStripFedCablingESProducer& operator=( const SiStripFedCablingESProducer& );
  
  virtual SiStripFedCabling* makeFedCabling() = 0; 
  
};

// ---------- inline methods ----------

uint16_t SiStripFedCablingESProducer::fecCrate( const uint16_t& module ) { return (module/(8*8*8*11)) % 4 + 1; }
uint16_t SiStripFedCablingESProducer::fecSlot( const uint16_t& module ) { return (module/(8*8*8)) % 11 + 2; }
uint16_t SiStripFedCablingESProducer::fecRing( const uint16_t& module ) { return (module/(8*8)) % 8 + 1; }
uint16_t SiStripFedCablingESProducer::ccuAddr( const uint16_t& module ) { return (module/8) % 8 + 1; }
uint16_t SiStripFedCablingESProducer::ccuChan( const uint16_t& module ) { return (module/1) % 8 + 16; }

#endif // CalibTracker_SiStripESProducers_SiStripFedCablingESProducer_H


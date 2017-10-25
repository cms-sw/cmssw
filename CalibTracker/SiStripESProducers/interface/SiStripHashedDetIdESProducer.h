#ifndef CalibTracker_SiStripESProducers_SiStripHashedDetIdESProducer_H
#define CalibTracker_SiStripESProducers_SiStripHashedDetIdESProducer_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "boost/cstdint.hpp"
#include <memory>

class SiStripHashedDetId;
class SiStripHashedDetIdRcd;

/** 
    @class SiStripHashedDetIdESProducer
    @author R.Bainbridge
    @brief Abstract base class for producer of SiStripHashedDetId record.
*/
class SiStripHashedDetIdESProducer : public edm::ESProducer {

 public:

  SiStripHashedDetIdESProducer( const edm::ParameterSet& );
  ~SiStripHashedDetIdESProducer() override;

  /** Calls pure virtual make() method, to force concrete implementation. */
  virtual std::unique_ptr<SiStripHashedDetId> produce( const SiStripHashedDetIdRcd& );
  
 private:
  
  SiStripHashedDetIdESProducer( const SiStripHashedDetIdESProducer& ) = delete;
  const SiStripHashedDetIdESProducer& operator=( const SiStripHashedDetIdESProducer& ) = delete;
  
  virtual SiStripHashedDetId* make( const SiStripHashedDetIdRcd& ) = 0; 
  
};

#endif // CalibTracker_SiStripESProducers_SiStripHashedDetIdESProducer_H


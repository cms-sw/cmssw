#ifndef CalibTracker_SiStripESProducers_SiStripHashedDetIdESProducer_h
#define CalibTracker_SiStripESProducers_SiStripHashedDetIdESProducer_h

#include "CalibFormats/SiStripObjects/interface/SiStripHashedDetId.h"
#include "CalibTracker/Records/interface/SiStripHashedDetIdRcd.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class SiStripHashedDetIdESProducer : public edm::ESProducer {

 public:

  SiStripHashedDetIdESProducer( const edm::ParameterSet& );
  
  virtual ~SiStripHashedDetIdESProducer();
  
  std::auto_ptr<SiStripHashedDetId> produce( const SiStripHashedDetIdRcd& );
  
};

#endif // CalibTracker_SiStripESProducers_SiStripHashedDetIdESProducer_h


#ifndef CalibTracker_SiStripESProducers_SiStripHashedDetIdESModule_h
#define CalibTracker_SiStripESProducers_SiStripHashedDetIdESModule_h

#include "CalibTracker/SiStripESProducers/interface/SiStripHashedDetIdESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class SiStripHashedDetId;
class SiStripHashedDetIdRcd;

/**
   @class SiStripHashedDetIdESModule
   @author R.Bainbridge
   @brief Builds hashed DetId map based on DetIds read from geometry database
*/
class SiStripHashedDetIdESModule : public SiStripHashedDetIdESProducer {
  
 public:
  
  SiStripHashedDetIdESModule( const edm::ParameterSet& );
  virtual ~SiStripHashedDetIdESModule();
  
 private:
  
  /** Builds hashed DetId map based on geometry. */
  virtual SiStripHashedDetId* make( const SiStripHashedDetIdRcd& ); 
  
};

#endif // CalibTracker_SiStripESProducers_SiStripHashedDetIdESModule_h


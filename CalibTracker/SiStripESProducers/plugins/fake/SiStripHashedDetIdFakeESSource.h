#ifndef CalibTracker_SiStripESProducers_SiStripHashedDetIdFakeESSource_H
#define CalibTracker_SiStripESProducers_SiStripHashedDetIdFakeESSource_H

#include "CalibTracker/SiStripESProducers/interface/SiStripHashedDetIdESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

class SiStripHashedDetId;
class SiStripHashedDetIdRcd;

/**
   @class SiStripHashedDetIdFakeESSource
   @author R.Bainbridge
   @brief Builds hashed DetId map based on list of DetIds read from ascii file
*/
class SiStripHashedDetIdFakeESSource : public SiStripHashedDetIdESProducer, public edm::EventSetupRecordIntervalFinder {
  
 public:
  
  explicit SiStripHashedDetIdFakeESSource( const edm::ParameterSet& );
  virtual ~SiStripHashedDetIdFakeESSource();
  
 protected:
  
  virtual void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
			       const edm::IOVSyncValue&,
			       edm::ValidityInterval& );
  
 private:
  
  /** Builds hashed DetId map based on ascii file. */
  virtual SiStripHashedDetId* make( const SiStripHashedDetIdRcd& ); 
  
  /** Location of ascii file containing DetIds. */
  edm::FileInPath detIds_;
  
};

#endif // CalibTracker_SiStripESProducers_SiStripHashedDetIdFakeESSource_H



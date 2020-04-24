#ifndef CalibTracker_SiStripESProducers_Phase2TrackerCablingCfgESSource_H
#define CalibTracker_SiStripESProducers_Phase2TrackerCablingCfgESSource_H

#include "CalibTracker/SiStripESProducers/interface/Phase2TrackerCablingESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class Phase2TrackerCabling;
class Phase2TrackerCablingRcd;

class Phase2TrackerCablingCfgESSource : public Phase2TrackerCablingESProducer, public edm::EventSetupRecordIntervalFinder {
  
 public:
  
  explicit Phase2TrackerCablingCfgESSource( const edm::ParameterSet& );
  ~Phase2TrackerCablingCfgESSource() override;
  
 protected:
  
  void setIntervalFor( const edm::eventsetup::EventSetupRecordKey& key,
			       const edm::IOVSyncValue& iov_sync,
			       edm::ValidityInterval& iov_validity ) override {
    edm::ValidityInterval infinity( iov_sync.beginOfTime(), iov_sync.endOfTime() );
    iov_validity = infinity;
  }

 private:
  
  // Builds cabling map based on the ParameterSet
  Phase2TrackerCabling* make( const Phase2TrackerCablingRcd& ) override; 

  // The configuration used to generated the cabling record
  edm::ParameterSet pset_;
};

#endif // CalibTracker_SiStripESProducers_Phase2TrackerCablingCfgESSource_H



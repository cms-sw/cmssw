#ifndef CalibTracker_SiStripESProducers_SiStripThresholdFakeESSource_H
#define CalibTracker_SiStripESProducers_SiStripThresholdFakeESSource_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/DataRecord/interface/SiStripThresholdRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripThreshold.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "boost/cstdint.hpp"
#include <memory>


/** 
    @class SiStripThresholdFakeESSource
    @brief Fake source of SiStripThreshold.
    @author D. Giordano
*/
class SiStripThresholdFakeESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {

 public:

  SiStripThresholdFakeESSource( const edm::ParameterSet& );
  virtual ~SiStripThresholdFakeESSource() {;}
  
  virtual std::auto_ptr<SiStripThreshold> produce( const SiStripThresholdRcd& );
  
  
 protected:
  
  virtual void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
			       const edm::IOVSyncValue&,
			       edm::ValidityInterval& );
  
 private:
  
  SiStripThresholdFakeESSource( const SiStripThresholdFakeESSource& );
  const SiStripThresholdFakeESSource& operator=( const SiStripThresholdFakeESSource& );

private:

  edm::FileInPath fp_;
  float lTh_, hTh_;
};


#endif // CalibTracker_SiStripThresholdESProducers_SiStripThresholdFakeESSource_H


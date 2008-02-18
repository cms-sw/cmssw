#ifndef CalibTracker_SiStripESProducers_SiStripGainFakeESSource_H
#define CalibTracker_SiStripESProducers_SiStripGainFakeESSource_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/DataRecord/interface/SiStripApvGainRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "boost/cstdint.hpp"
#include <memory>


/** 
    @class SiStripGainFakeESSource
    @brief Fake source of SiStripApvGain.
    @author G. Bruno
*/
class SiStripGainFakeESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {

 public:

  SiStripGainFakeESSource( const edm::ParameterSet& );
  virtual ~SiStripGainFakeESSource() {;}
  
  virtual std::auto_ptr<SiStripApvGain> produce( const SiStripApvGainRcd& );
  
  
 protected:
  
  virtual void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
			       const edm::IOVSyncValue&,
			       edm::ValidityInterval& );
  
 private:
  
  SiStripGainFakeESSource( const SiStripGainFakeESSource& );
  const SiStripGainFakeESSource& operator=( const SiStripGainFakeESSource& );

private:

  edm::FileInPath fp_;

};


#endif // CalibTracker_SiStripGainESProducers_SiStripGainFakeESSource_H


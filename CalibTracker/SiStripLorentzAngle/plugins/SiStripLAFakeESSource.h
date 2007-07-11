#ifndef CalibTracker_SiStripLorentzAngle_SiStripLAFakeESSource_H
#define CalibTracker_SiStripLorentzAngle_SiStripLAFakeESSource_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/DataRecord/interface/SiStripLorentzAngleRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "boost/cstdint.hpp"
#include <memory>


/** 
    @class SiStripLAFakeESSource
    @brief Fake source of SiStripLorentzAngle.
    @author C. Genta
*/
class SiStripLAFakeESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {

 public:

  SiStripLAFakeESSource( const edm::ParameterSet& );
  virtual ~SiStripLAFakeESSource() {;}
  
  virtual std::auto_ptr<SiStripLorentzAngle> produce( const SiStripLorentzAngleRcd& );
  
  
 protected:
  
  virtual void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
			       const edm::IOVSyncValue&,
			       edm::ValidityInterval& );
  
 private:
  
  SiStripLAFakeESSource( const SiStripLAFakeESSource& );
  const SiStripLAFakeESSource& operator=( const SiStripLAFakeESSource& );


  edm::FileInPath fp_;
  double appliedVoltage_;
  double chargeMobility_;
  double temperature_;
  double temperatureerror_;
  double rhall_;
  double holeBeta_;
  double holeSaturationVelocity_;
};


#endif // CalibTracker_SiStripLorentzAngle_SiStripLAFakeESSource_H


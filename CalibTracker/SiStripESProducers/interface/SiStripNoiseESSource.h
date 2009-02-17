// Last commit: $Id: SiStripNoiseESSource.h,v 1.2 2008/12/17 23:11:13 giordano Exp $
// Latest tag:  $Name: V03-00-00-00 $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/CalibTracker/SiStripESProducers/interface/SiStripNoiseESSource.h,v $

#ifndef CalibTracker_SiStripESProducers_SiStripNoiseESSource_H
#define CalibTracker_SiStripESProducers_SiStripNoiseESSource_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "boost/cstdint.hpp"
#include <memory>

class SiStripNoises;
class SiStripNoisesRcd;

/** 
    @class SiStripNoiseESSource
    @brief Pure virtual class for EventSetup sources of SiStripNoises.
    @author R.Bainbridge
*/
class SiStripNoiseESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {

 public:

  SiStripNoiseESSource( const edm::ParameterSet& );
  virtual ~SiStripNoiseESSource() {;}
  
  virtual std::auto_ptr<SiStripNoises> produce( const SiStripNoisesRcd& );
  
 protected:

  virtual void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
			       const edm::IOVSyncValue&,
			       edm::ValidityInterval& );
  
 private:
  
  SiStripNoiseESSource( const SiStripNoiseESSource& );
  const SiStripNoiseESSource& operator=( const SiStripNoiseESSource& );

  virtual SiStripNoises* makeNoise() = 0; 

};

#endif // CalibTracker_SiStripESProducers_SiStripNoiseESSource_H



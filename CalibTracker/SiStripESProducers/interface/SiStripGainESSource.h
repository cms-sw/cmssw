<<<<<<< SiStripGainESSource.h
// Last commit: $Id: SiStripGainESSource.h,v 1.2 2008/12/17 23:11:13 giordano Exp $
// Latest tag:  $Name: V03-00-04 $
=======
// Last commit: $Id: SiStripGainESSource.h,v 1.3 2009/02/17 16:14:33 muzaffar Exp $
// Latest tag:  $Name:  $
>>>>>>> 1.3
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/CalibTracker/SiStripESProducers/interface/SiStripGainESSource.h,v $

#ifndef CalibTracker_SiStripESProducers_SiStripGainESSource_H
#define CalibTracker_SiStripESProducers_SiStripGainESSource_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <memory>

class SiStripApvGain;
class SiStripApvGainRcd;

/** 
    @class SiStripGainESSource
    @brief Pure virtual class for EventSetup sources of SiStripApvGain.
    @author R.Bainbridge
*/
class SiStripGainESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {

 public:

  SiStripGainESSource( const edm::ParameterSet& );
  virtual ~SiStripGainESSource() {;}
  
  virtual std::auto_ptr<SiStripApvGain> produce( const SiStripApvGainRcd& );
  
 protected:

  virtual void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
			       const edm::IOVSyncValue&,
			       edm::ValidityInterval& );
  
 private:
  
  SiStripGainESSource( const SiStripGainESSource& );
  const SiStripGainESSource& operator=( const SiStripGainESSource& );

  virtual SiStripApvGain* makeGain() = 0; 

};

#endif // CalibTracker_SiStripESProducers_SiStripGainESSource_H



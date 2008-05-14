// Last commit: $Id: SiStripPedestalsESSource.h,v 1.1 2006/12/22 12:04:33 bainbrid Exp $
// Latest tag:  $Name: V07-00-02-00 $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/CalibTracker/SiStripPedestals/interface/SiStripPedestalsESSource.h,v $

#ifndef CalibTracker_SiStripESProducers_SiStripPedestalsESSource_H
#define CalibTracker_SiStripESProducers_SiStripPedestalsESSource_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "boost/cstdint.hpp"
#include <memory>

class SiStripPedestals;
class SiStripPedestalsRcd;

/** 
    @class SiStripPedestalsESSource
    @brief Pure virtual class for EventSetup sources of SiStripPedestals.
    @author R.Bainbridge
*/
class SiStripPedestalsESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {

 public:

  SiStripPedestalsESSource( const edm::ParameterSet& );
  virtual ~SiStripPedestalsESSource() {;}
  
  virtual std::auto_ptr<SiStripPedestals> produce( const SiStripPedestalsRcd& );
  
 protected:

  virtual void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
			       const edm::IOVSyncValue&,
			       edm::ValidityInterval& );
  
 private:
  
  SiStripPedestalsESSource( const SiStripPedestalsESSource& );
  const SiStripPedestalsESSource& operator=( const SiStripPedestalsESSource& );

  virtual SiStripPedestals* makePedestals() = 0; 

};

#endif // CalibTracker_SiStripESProducers_SiStripPedestalsESSource_H



#ifndef CalibTracker_SiStripESProducers_SiStripFedCablingFakeESSource_H
#define CalibTracker_SiStripESProducers_SiStripFedCablingFakeESSource_H

#include "CalibTracker/SiStripESProducers/interface/SiStripFedCablingESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

class SiStripFedCabling;
class SiStripFedCablingRcd;

/**
   @class SiStripFedCablingFakeESSource
   @author R.Bainbridge
   @brief Builds cabling map based on list of DetIds and FedIds read from ascii files
*/
class SiStripFedCablingFakeESSource : public SiStripFedCablingESProducer, public edm::EventSetupRecordIntervalFinder {
  
 public:
  
  explicit SiStripFedCablingFakeESSource( const edm::ParameterSet& );
  ~SiStripFedCablingFakeESSource();
  
 protected:
  
  virtual void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
			       const edm::IOVSyncValue&,
			       edm::ValidityInterval& );
  
 private:
  
  /** Builds cabling map based on ascii files. */
  virtual SiStripFedCabling* make( const SiStripFedCablingRcd& ); 

  /** Location of ascii file containing DetIds. */
  edm::FileInPath detIds_;

  /** Location of ascii file containing FedIds. */
  edm::FileInPath fedIds_;
  edm::ParameterSet pset_;
};

#endif // CalibTracker_SiStripESProducers_SiStripFedCablingFakeESSource_H



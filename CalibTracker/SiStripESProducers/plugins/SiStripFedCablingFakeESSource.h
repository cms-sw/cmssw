#ifndef CalibTracker_SiStripConnectivity_SiStripFedCablingTrivialBuilder_H
#define CalibTracker_SiStripConnectivity_SiStripFedCablingTrivialBuilder_H

#include "CalibTracker/SiStripESProducers/plugins/SiStripFedCablingESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

class SiStripFedCabling;

/**
   @class SiStripFedCablingFakeESSource
   @author R.Bainbridge
*/
class SiStripFedCablingFakeESSource : public SiStripFedCablingESProducer {
  
 public:
  
  explicit SiStripFedCablingFakeESSource( const edm::ParameterSet& );
  ~SiStripFedCablingFakeESSource();
  
 private:
  
  /** Builds "fake" cabling map, based on real DetIds and FedIds. */
  virtual SiStripFedCabling* makeFedCabling(); 

  edm::FileInPath detIds_;

  edm::FileInPath fedIds_;
  
};

#endif // CalibTracker_SiStripConnectivity_SiStripFedCablingTrivialBuilder_H



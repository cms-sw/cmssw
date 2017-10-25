
#ifndef CALIBTRACKER_SISTRIPCONNECTIVITY_SISTRIPREGIONCONNECTIVITY_H
#define CALIBTRACKER_SISTRIPCONNECTIVITY_SISTRIPREGIONCONNECTIVITY_H

// FWCore
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// CalibTracker
#include "CalibTracker/Records/interface/SiStripRegionCablingRcd.h"

// CalibFormats
#include "CalibFormats/SiStripObjects/interface/SiStripRegionCabling.h"

class SiStripRegionConnectivity: public edm::ESProducer {

 public:

  SiStripRegionConnectivity( const edm::ParameterSet& );
  ~SiStripRegionConnectivity() override;
  
  std::unique_ptr<SiStripRegionCabling> produceRegionCabling( const SiStripRegionCablingRcd&  );
  
 private:

  /** Number of regions in eta,phi */
  uint32_t etadivisions_;
  uint32_t phidivisions_;

  /** Tracker extent in eta */
  double etamax_;
};

#endif


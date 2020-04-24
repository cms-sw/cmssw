#ifndef CALIBTRACKER_SISTRIPCONNECTIVITY_SISTRIPCONNECTIVITY_H
#define CALIBTRACKER_SISTRIPCONNECTIVITY_SISTRIPCONNECTIVITY_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CalibTracker/Records/interface/SiStripFecCablingRcd.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

class SiStripConnectivity: public edm::ESProducer {

 public:
  
  SiStripConnectivity( const edm::ParameterSet& );
  ~SiStripConnectivity() override;
  
  std::unique_ptr<SiStripFecCabling> produceFecCabling( const SiStripFecCablingRcd& );
  std::unique_ptr<SiStripDetCabling> produceDetCabling( const SiStripDetCablingRcd& );
  
 private:
  
};

#endif

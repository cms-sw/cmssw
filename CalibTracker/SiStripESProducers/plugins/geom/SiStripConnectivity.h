// Last commit: $Id: SiStripConnectivity.h,v 1.3 2008/04/02 12:13:08 bainbrid Exp $
// Latest tag:  $Name: V04-02-14 $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/CalibTracker/SiStripConnectivity/plugins/SiStripConnectivity.h,v $

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
  virtual ~SiStripConnectivity();
  
  std::auto_ptr<SiStripFecCabling> produceFecCabling( const SiStripFecCablingRcd& );
  std::auto_ptr<SiStripDetCabling> produceDetCabling( const SiStripDetCablingRcd& );
  
 private:
  
};

#endif

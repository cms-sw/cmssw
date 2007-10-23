#ifndef EventFilter_SiStripRawToDigi_SiStripFEDRawDataCheck_H
#define EventFilter_SiStripRawToDigi_SiStripFEDRawDataCheck_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <string>

class SiStripRawToDigiUnpacker;

/**
   @class SiStripFEDRawDataCheck 
   @brief Analyzes contents of FEDRawData collection
*/
class SiStripFEDRawDataCheck {
  
 public:
  
  SiStripFEDRawDataCheck( const edm::ParameterSet& );
  virtual ~SiStripFEDRawDataCheck();
  
  void analyze( const edm::Event&, const edm::EventSetup& );
  
 private:
  
  SiStripFEDRawDataCheck() {;}
  
  std::string label_;
  std::string instance_;
  
  SiStripRawToDigiUnpacker* unpacker_;
  
};

#endif // EventFilter_SiStripRawToDigi_SiStripFEDRawDataCheck_H


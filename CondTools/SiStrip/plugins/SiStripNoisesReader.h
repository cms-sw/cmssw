#ifndef SiStripNoisesReader_H
#define SiStripNoisesReader_H

// system include files
//#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"

#include <iostream>
#include <stdio.h>
#include <sys/time.h>

class SiStripNoisesReader : public edm::EDAnalyzer {

 public:
  explicit SiStripNoisesReader( const edm::ParameterSet& );
  ~SiStripNoisesReader();
  
  void analyze( const edm::Event&, const edm::EventSetup& );

 private:
  uint32_t printdebug_;
};
#endif

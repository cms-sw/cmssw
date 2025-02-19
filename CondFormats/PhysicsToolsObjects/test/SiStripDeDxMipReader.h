#ifndef SiStripDeDxMipReader_H
#define SiStripDeDxMipReader_H

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



class SiStripDeDxMipReader : public edm::EDAnalyzer {

 public:
  explicit SiStripDeDxMipReader( const edm::ParameterSet& );
  ~SiStripDeDxMipReader();
  
  void analyze( const edm::Event&, const edm::EventSetup& );

 private:
  //  uint32_t printdebug_;

};
#endif

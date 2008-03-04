#ifndef SiStripFakeRawDigiModule_H
#define SiStripFakeRawDigiModule_H

#include "FWCore/Framework/interface/EDProducer.h"
#include <string>


class SiStripFakeRawDigiModule : public edm::EDProducer {
  
 public:
  
  SiStripFakeRawDigiModule( const edm::ParameterSet& );
  ~SiStripFakeRawDigiModule(){};

  virtual void beginJob( const edm::EventSetup& ) {;}
  virtual void endJob() {;}
  
  virtual void produce( edm::Event&, const edm::EventSetup& );
  
};

#endif 

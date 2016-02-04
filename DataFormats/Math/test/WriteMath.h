#ifndef DataFormats_WriteMath_h
#define DataFormats_WriteMath_h
#include "FWCore/Framework/interface/EDProducer.h"

class WriteMath : public edm::EDProducer {
public:
  WriteMath( const edm::ParameterSet& );
private:
  void produce( edm::Event &, const edm::EventSetup & );
};

#endif

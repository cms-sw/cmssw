
#ifndef GsfElectronProducer_h
#define GsfElectronProducer_h

class GsfElectronAlgo ;

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

namespace edm
 {
  class ParameterSet ;
  class ConfigurationDescriptions ;
 }

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/EDProduct.h"

class GsfElectronProducer : public edm::EDProducer
 {
  public:

    //static void fillDescriptions( edm::ConfigurationDescriptions & ) ;

    explicit GsfElectronProducer( const edm::ParameterSet & ) ;
    virtual ~GsfElectronProducer();

    virtual void produce( edm::Event &, const edm::EventSetup & ) ;

  private:

    GsfElectronAlgo * algo_ ;

 } ;

#endif

#ifndef GsfElectronCoreProducer_h
#define GsfElectronCoreProducer_h

//
// Package:         RecoEgamma/EgammaElectronProducers
// Class:           GsfElectronCoreProducer
//
// Description:


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class GsfElectronCoreProducer : public edm::EDProducer
 {
  public:
    explicit GsfElectronCoreProducer( const edm::ParameterSet & conf ) ;
    virtual ~GsfElectronCoreProducer() ;
    virtual void produce( edm::Event& e, const edm::EventSetup & c ) ;
  private:
	edm::InputTag gsfTracksTag_ ;
	edm::InputTag pfSuperClustersTag_ ;
 } ;


#endif

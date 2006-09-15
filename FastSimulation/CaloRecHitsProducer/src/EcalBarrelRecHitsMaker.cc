#include "FastSimulation/CaloRecHitsProducer/interface/EcalBarrelRecHitsMaker.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include <iostream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

EcalBarrelRecHitsMaker::EcalBarrelRecHitsMaker(edm::ParameterSet const & p)
{
  edm::ParameterSet RecHitsParameters = p.getParameter<edm::ParameterSet>("ECALBarrel");
  noise_ = RecHitsParameters.getParameter<double>("Noise");
  threshold_ = RecHitsParameters.getParameter<double>("Threshold");
  barrelhits_.resize(61200,0.);
}

EcalBarrelRecHitsMaker::~EcalBarrelRecHitsMaker()
{;
}


void EcalBarrelRecHitsMaker::loadEcalBarrelRecHits(edm::Event &iEvent,EBRecHitCollection & ecalHits)
{
  edm::Handle<edm::PCaloHitContainer> pcalohits;
  iEvent.getByLabel("Famos","EcalHitsEB",pcalohits);

  edm::PCaloHitContainer::const_iterator it=pcalohits.product()->begin();
  edm::PCaloHitContainer::const_iterator itend=pcalohits.product()->end();
  unsigned counter=0;
  for(;it!=itend;++it)
    {
      ecalHits.push_back(EcalRecHit(DetId(it->id()),it->energy(),0.)); 
      ++counter;
    }
  //  std::cout << " Ecal Barrel. RecHits created " << counter << std::endl;
}

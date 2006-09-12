#include "FastSimulation/CaloRecHitsProducer/interface/EcalEndcapRecHitsMaker.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include <iostream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

EcalEndcapRecHitsMaker::EcalEndcapRecHitsMaker(edm::ParameterSet const & p)
{
  edm::ParameterSet RecHitsParameters = p.getParameter<edm::ParameterSet>("ECALEndcap");
  noise_ = RecHitsParameters.getParameter<double>("Noise");
  threshold_ = RecHitsParameters.getParameter<double>("Threshold");
}

EcalEndcapRecHitsMaker::~EcalEndcapRecHitsMaker()
{;
}


void EcalEndcapRecHitsMaker::loadEcalEndcapRecHits(edm::Event &iEvent,EERecHitCollection & ecalHits)
{
  edm::Handle<edm::PCaloHitContainer> pcalohits;
  iEvent.getByLabel("Famos","EcalHitsEE",pcalohits);

  edm::PCaloHitContainer::const_iterator it=pcalohits.product()->begin();
  edm::PCaloHitContainer::const_iterator itend=pcalohits.product()->end();
  unsigned counter=0;
  for(;it!=itend;++it)
    {
      ecalHits.push_back(EcalRecHit(DetId(it->id()),it->energy(),0.)); 
      ++counter;
    }
  //  std::cout << " Ecal Endcap. RecHits created " << counter << std::endl;
}

#include "FastSimulation/CaloRecHitsProducer/interface/HcalRecHitsMaker.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include <iostream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

HcalRecHitsMaker::HcalRecHitsMaker(edm::ParameterSet const & p):initialized_(false)
{
  edm::ParameterSet RecHitsParameters = p.getParameter<edm::ParameterSet>("HCAL");
  noise_ = RecHitsParameters.getParameter<double>("Noise");
  threshold_ = RecHitsParameters.getParameter<double>("Threshold");
}

HcalRecHitsMaker::~HcalRecHitsMaker()
{;
}

void HcalRecHitsMaker::init(const edm::EventSetup &es)
{
  if(initialized_) return;
  // es.get<IdealGeometryRecord>().get(calotowerMap_);
  initialized_=true;
}

void HcalRecHitsMaker::loadHcalRecHits(edm::Event &iEvent,HBHERecHitCollection& hbheHits, HORecHitCollection &hoHits,HFRecHitCollection &hfHits)
{
  edm::Handle<edm::PCaloHitContainer> pcalohits;
  iEvent.getByLabel("Famos","HcalHits",pcalohits);

  edm::PCaloHitContainer::const_iterator it=pcalohits.product()->begin();
  edm::PCaloHitContainer::const_iterator itend=pcalohits.product()->end();
  unsigned counter=0;
  for(;it!=itend;++it)
    {
      HcalDetId detid(it->id());

      //      CaloTowerDetId mytowerId=calotowerMap_->towerOf(detid);
      //      std::cout << " Found CaloTower " << detid << " " << mytowerId << std::endl;

      switch(detid.subdet())
	{
	case HcalBarrel: 
	  hbheHits.push_back(HBHERecHit(detid,it->energy(),0)); 
	  break;
	case HcalEndcap: 
	  hbheHits.push_back(HBHERecHit(detid,it->energy(),0)); 
	  break;
	case HcalOuter: 
	  hoHits.push_back(HORecHit(detid,it->energy(),0));
	  break;		     
	case HcalForward: 
	  hfHits.push_back(HFRecHit(detid,it->energy(),0));
	  break;
	default:
	  edm::LogWarning("FastCalorimetry") << "RecHit not registered\n";
	  ;
	}
      ++counter;
    }
  //  std::cout << " Hcal RecHits created " << counter << std::endl;
}

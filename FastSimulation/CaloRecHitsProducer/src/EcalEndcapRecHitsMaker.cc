#include "FastSimulation/CaloRecHitsProducer/interface/EcalEndcapRecHitsMaker.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CLHEP/Random/RandGaussQ.h"
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

void EcalEndcapRecHitsMaker::clean()
{

  ecaleRecHits_.clear();
}


void EcalEndcapRecHitsMaker::loadEcalEndcapRecHits(edm::Event &iEvent,EERecHitCollection & ecalHits)
{


  this->clean();

  loadPSimHits(iEvent);

  std::map<uint32_t,float>::const_iterator it=ecaleRecHits_.begin();
  std::map<uint32_t,float>::const_iterator itend=ecaleRecHits_.end();

  for(;it!=itend;++it)
    {

      ecalHits.push_back(EcalRecHit(EEDetId(it->first),it->second,0.)); 
    }
}

void EcalEndcapRecHitsMaker::loadPSimHits(const edm::Event & iEvent)
{

  edm::Handle<edm::PCaloHitContainer> pcalohits;
  iEvent.getByLabel("Famos","EcalHitsEE",pcalohits);

  edm::PCaloHitContainer::const_iterator it=pcalohits.product()->begin();
  edm::PCaloHitContainer::const_iterator itend=pcalohits.product()->end();
  for(;it!=itend;++it)
    {
      EEDetId detid(it->id());      
      noisifyAndFill(detid.rawId(),it->energy(),ecaleRecHits_);
    }

}

// Takes a hit (from a PSimHit) and fills a map with it after adding the noise. 
void EcalEndcapRecHitsMaker::noisifyAndFill(uint32_t id,float energy, std::map<uint32_t,float>& myHits)
{


  if (noise_>0.) energy +=   RandGaussQ::shoot(0.,noise_);

  // If below the threshold, a hit is nevertheless created, otherwise, there is a risk that a "noisy" hit 
  // is afterwards put in this cell which would not be correct. 
  if ( energy <threshold_ ) energy=0.;
  // No double counting check. Depending on how the pile-up is implemented , this can be a problem.
  // Fills the map giving a "hint", in principle, the objets have already been ordered in CalorimetryManager
  myHits.insert(myHits.end(),std::pair<uint32_t,float>(id,energy));
}

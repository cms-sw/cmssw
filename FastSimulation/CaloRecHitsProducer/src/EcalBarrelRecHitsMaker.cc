#include "FastSimulation/CaloRecHitsProducer/interface/EcalBarrelRecHitsMaker.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "FastSimulation/Utilities/interface/RandomEngine.h"
#include <iostream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

EcalBarrelRecHitsMaker::EcalBarrelRecHitsMaker(edm::ParameterSet const & p,const RandomEngine* myrandom):random_(myrandom)
{
  edm::ParameterSet RecHitsParameters = p.getParameter<edm::ParameterSet>("ECALBarrel");
  noise_ = RecHitsParameters.getParameter<double>("Noise");
  threshold_ = RecHitsParameters.getParameter<double>("Threshold");
}

EcalBarrelRecHitsMaker::~EcalBarrelRecHitsMaker()
{;
}


void EcalBarrelRecHitsMaker::clean()
{
  ecalbRecHits_.clear();
}

void EcalBarrelRecHitsMaker::loadEcalBarrelRecHits(edm::Event &iEvent,EBRecHitCollection & ecalHits)
{

  clean();
  loadPSimHits(iEvent);
  
  std::map<uint32_t,float>::const_iterator it=ecalbRecHits_.begin();
  std::map<uint32_t,float>::const_iterator itend=ecalbRecHits_.end();
  
  for(;it!=itend;++it)
    {
      ecalHits.push_back(EcalRecHit(EBDetId(it->first),it->second,0.));
    }
}

void EcalBarrelRecHitsMaker::loadPSimHits(const edm::Event & iEvent)
{

  edm::Handle<edm::PCaloHitContainer> pcalohits;
  iEvent.getByLabel("Famos","EcalHitsEB",pcalohits);

  edm::PCaloHitContainer::const_iterator it=pcalohits.product()->begin();
  edm::PCaloHitContainer::const_iterator itend=pcalohits.product()->end();
  for(;it!=itend;++it)
    {
      EBDetId detid(it->id());      
      noisifyAndFill(detid.rawId(),it->energy(),ecalbRecHits_);
    }
}

// Takes a hit (from a PSimHit) and fills a map with it after adding the noise. 
void EcalBarrelRecHitsMaker::noisifyAndFill(uint32_t id,float energy, std::map<uint32_t,float>& myHits)
{


  if (noise_>0.) energy +=   random_->gaussShoot(0.,noise_);

  // If the energy+noise is below the threshold, a hit is nevertheless created, otherwise, there is a risk that a "noisy" hit 
  // is afterwards put in this cell which would not be correct. 
  if ( energy <threshold_ ) energy=0.;
  // No double counting check. Depending on how the pile-up is implemented , this can be a problem.
  // Fills the map giving a "hint", in principle, the objets have already been ordered in CalorimetryManager
  myHits.insert(myHits.end(),std::pair<uint32_t,float>(id,energy));
}

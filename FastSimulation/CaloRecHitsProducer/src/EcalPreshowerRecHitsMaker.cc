#include "FastSimulation/CaloRecHitsProducer/interface/EcalPreshowerRecHitsMaker.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "FastSimulation/Utilities/interface/RandomEngine.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include <iostream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CLHEP/GenericFunctions/Erf.hh"

EcalPreshowerRecHitsMaker::EcalPreshowerRecHitsMaker(edm::ParameterSet const & p, const RandomEngine * myrandom):random_(myrandom)
{
  edm::ParameterSet RecHitsParameters = p.getParameter<edm::ParameterSet>("ECALPreshower");
  noise_ = RecHitsParameters.getParameter<double>("Noise");
  threshold_ = RecHitsParameters.getParameter<double>("Threshold");
  initialized_=false;

  Genfun::Erf myErf;
  if(noise_>0.)
    preshowerHotFraction_ = 0.5-0.5*myErf(threshold_/noise_/sqrt(2.));
  else
    preshowerHotFraction_ =0.;

  if(noise_>0.) myGaussianTailGenerator_.setParameters(noise_,threshold_);
}

EcalPreshowerRecHitsMaker::~EcalPreshowerRecHitsMaker()
{
  initialized_=false;
}

void EcalPreshowerRecHitsMaker::clean()
{

  ecalsRecHits_.clear();
}


void EcalPreshowerRecHitsMaker::loadEcalPreshowerRecHits(edm::Event &iEvent,ESRecHitCollection & ecalHits)
{

  loadPSimHits(iEvent);
  if(noise_>0.) noisifySubdet(ecalsRecHits_,escells_,ncells_);

  std::map<SignalHit,float>::const_iterator it=ecalsRecHits_.begin();
  std::map<SignalHit,float>::const_iterator itend=ecalsRecHits_.end();

  for(;it!=itend;++it)
    {
      
      if(it->first.killed()) continue;
      ESDetId detid(it->first.id());
      ecalHits.push_back(EcalRecHit(detid,it->second,0.)); 
    }
}

void EcalPreshowerRecHitsMaker::loadPSimHits(const edm::Event & iEvent)
{

  clean();

  edm::Handle<edm::PCaloHitContainer> pcalohits;
  iEvent.getByLabel("Famos","EcalHitsES",pcalohits);

  edm::PCaloHitContainer::const_iterator it=pcalohits.product()->begin();
  edm::PCaloHitContainer::const_iterator itend=pcalohits.product()->end();
  for(;it!=itend;++it)
    {
      noisifyAndFill(it->id(),it->energy(),ecalsRecHits_);
    }
}

void EcalPreshowerRecHitsMaker::init(const edm::EventSetup &es)
{
  if(initialized_) return;
  ncells_=createVectorsOfCells(es);
  edm::LogInfo("CaloRecHitsProducer") << "Total number of cells in Preshower " << ncells_ << std::endl;
  initialized_=true;
}

unsigned EcalPreshowerRecHitsMaker::createVectorsOfCells(const edm::EventSetup &es)
{
    edm::ESHandle<CaloGeometry> pG;
    es.get<IdealGeometryRecord>().get(pG);     
    unsigned total=0;

    escells.reserve(137728);

    const CaloSubdetectorGeometry* geom=pG->getSubdetectorGeometry(DetId::Ecal,EcalPreshower);  
    std::vector<DetId> ids=geom->getValidDetIds(DetId::Ecal,EcalPreshower);  
    for (std::vector<DetId>::iterator i=ids.begin(); i!=ids.end(); i++) 
      {
	escells_.push_back(i->rawId());
	++total;
      }
    return escells_.size();
}

void EcalPreshowerRecHitsMaker::noisifyAndFill(uint32_t id,float energy, std::map<SignalHit,float>& myHits)
{
  bool killed=false;
  // No double counting check. Depending on how the pile-up is implemented , this can be a problem.

  if (noise_>0.) energy +=  random_->gaussShoot(0.,noise_);

  // If below the threshold, a hit is nevertheless created, otherwise, there is a risk that a "noisy" hit 
  // is afterwards put in this cell which would not be correct. 
  if ( energy <threshold_ ) 
    {
      energy=0.;
      killed=true;
    }
  // In principe (without pile-up), the hits have been already ordered, gives a "hint" to the insert
  myHits.insert(myHits.end(),std::pair<SignalHit,float>(SignalHit(id,killed),energy));
}


void EcalPreshowerRecHitsMaker::noisifySubdet(std::map<SignalHit,float>& theMap, const std::vector<uint32_t>& thecells, unsigned ncells)
{
  // noise won't be injected in cells that contain signal
  unsigned mean=(unsigned)((double)(ncells-theMap.size())*preshowerHotFraction_);
  unsigned nps = (unsigned)(random_->poissonShoot(mean));
  
  unsigned ncell=0;
  unsigned cellindex=0;
  uint32_t cellnumber=0;
  std::map<SignalHit,float>::const_iterator itcheck;

  while(ncell < nps)
    {
      cellindex = (unsigned)(random_->flatShoot()*ncells);
      cellnumber = thecells[cellindex];
      itcheck=theMap.find(SignalHit(cellnumber));
      if(itcheck==theMap.end()) // inject only in empty cells
	{
	  theMap.insert(std::pair<SignalHit,float>(SignalHit(cellnumber),myGaussianTailGenerator_.shoot()));
	  ++ncell;
	}
    }
  //  edm::LogInfo("CaloRecHitsProducer") << "CaloRecHitsProducer : added noise in "<<  ncell << " HCAL cells "  << std::endl;
}

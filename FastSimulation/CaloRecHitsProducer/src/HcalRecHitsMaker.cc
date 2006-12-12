#include "FastSimulation/CaloRecHitsProducer/interface/HcalRecHitsMaker.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "CLHEP/GenericFunctions/Erf.hh"

#include "FastSimulation/Utilities/interface/RandomEngine.h"

#include <algorithm>
#include <iostream>

class RandomEngine;

HcalRecHitsMaker::HcalRecHitsMaker(edm::ParameterSet const & p,const RandomEngine * myrandom):initialized_(false),random_(myrandom)
{
  edm::ParameterSet RecHitsParameters = p.getParameter<edm::ParameterSet>("HCAL");
  noise_ = RecHitsParameters.getParameter<double>("Noise");
  threshold_ = RecHitsParameters.getParameter<double>("Threshold");

  // Computes the fraction of HCAL above the threshold
  Genfun::Erf myErf;
  if(noise_>0.)
    hcalHotFraction_ = 0.5-0.5*myErf(threshold_/noise_/sqrt(2.));
  else
    hcalHotFraction_ =0.;

  if(noise_>0.) myGaussianTailGenerator_.setParameters(noise_,threshold_);
}

HcalRecHitsMaker::~HcalRecHitsMaker()
{;
}

void HcalRecHitsMaker::init(const edm::EventSetup &es)
{
  if(initialized_) return;
  unsigned ncells=createVectorsOfCells(es);
  edm::LogInfo("CaloRecHitsProducer") << "Total number of cells in HCAL " << ncells << std::endl;
  nhbhecells_ = hbhecells_.size();
  nhocells_ = hocells_.size();
  nhfcells_ = hfcells_.size(); 
  initialized_=true;
}

void HcalRecHitsMaker::loadPSimHits(const edm::Event & iEvent)
{

  clean();

  edm::Handle<edm::PCaloHitContainer> pcalohits;
  iEvent.getByLabel("Famos","HcalHits",pcalohits);

  edm::PCaloHitContainer::const_iterator it=pcalohits.product()->begin();
  edm::PCaloHitContainer::const_iterator itend=pcalohits.product()->end();
  unsigned counter=0;
  for(;it!=itend;++it)
    {
      HcalDetId detid(it->id());

      switch(detid.subdet())
	{
	case HcalBarrel: 
	  noisifyAndFill(detid.rawId(),it->energy(),hbheRecHits_);
	  break;
	case HcalEndcap: 
	  noisifyAndFill(detid.rawId(),it->energy(),hbheRecHits_);
	  break;
	case HcalOuter: 
	  noisifyAndFill(detid.rawId(),it->energy(),hoRecHits_);
	  break;		     
	case HcalForward: 
	  noisifyAndFill(detid.rawId(),it->energy(),hfRecHits_);
	  break;
	default:
	  edm::LogWarning("CaloRecHitsProducer") << "RecHit not registered\n";
	  ;
	}
      ++counter;
    }
}

// Fills the collections. 
void HcalRecHitsMaker::loadHcalRecHits(edm::Event &iEvent,HBHERecHitCollection& hbheHits, HORecHitCollection &hoHits,HFRecHitCollection &hfHits)
{

  loadPSimHits(iEvent);
  if (noise_>0.) noisify();

  // HB-HE
  std::map<signalHit,float>::const_iterator it=hbheRecHits_.begin();
  std::map<signalHit,float>::const_iterator itend=hbheRecHits_.end();
  
  for(;it!=itend;++it)
    {
      if(it->first.killed()) continue;
      HcalDetId detid(it->first.id());
      hbheHits.push_back(HBHERecHit(detid,it->second,0));
    }

  // HO
  it = hoRecHits_.begin();
  itend = hoRecHits_.end();
  for(;it!=itend;++it)
    {
      if(it->first.killed()) continue;
      HcalDetId detid(it->first.id());
      hoHits.push_back(HORecHit(detid,it->second,0));
    }
  
  // HF
  it = hfRecHits_.begin();
  itend = hfRecHits_.end();
  for(;it!=itend;++it)
    {
      if(it->first.killed()) continue;
      HcalDetId detid(it->first.id());
      hfHits.push_back(HFRecHit(detid,it->second,0));
    }

}


// For a fast injection of the noise: the list of cell ids is stored
unsigned HcalRecHitsMaker::createVectorsOfCells(const edm::EventSetup &es)
{
    edm::ESHandle<CaloGeometry> pG;
    es.get<IdealGeometryRecord>().get(pG);     
    unsigned total=0;
    total += createVectorOfSubdetectorCells(*pG,HcalBarrel,hbhecells_);
    total += createVectorOfSubdetectorCells(*pG,HcalEndcap,hbhecells_);
    total += createVectorOfSubdetectorCells(*pG,HcalOuter,hocells_);
    total += createVectorOfSubdetectorCells(*pG,HcalForward,hfcells_);    
    return total;
}

// list of the cellids for a given subdetector
unsigned HcalRecHitsMaker::createVectorOfSubdetectorCells(const CaloGeometry& cg,int subdetn,std::vector<uint32_t>& cellsvec ) 
{
  const CaloSubdetectorGeometry* geom=cg.getSubdetectorGeometry(DetId::Hcal,subdetn);  
  std::vector<DetId> ids=geom->getValidDetIds(DetId::Hcal,subdetn);  
  for (std::vector<DetId>::iterator i=ids.begin(); i!=ids.end(); i++) 
    {
      cellsvec.push_back(i->rawId());
    }
  return cellsvec.size();
}

// Takes a hit (from a PSimHit) and fills a map with it after adding the noise. 
void HcalRecHitsMaker::noisifyAndFill(uint32_t id,float energy, std::map<signalHit,float>& myHits)
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
  myHits.insert(myHits.end(),std::pair<signalHit,float>(signalHit(id,killed),energy));
}

void HcalRecHitsMaker::noisify()
{
  if(hbheRecHits_.size()<nhbhecells_)
    noisifySubdet(hbheRecHits_,hbhecells_,nhbhecells_);
  else
    edm::LogWarning("CaloRecHitsProducer") << "All HCAL(HB-HE) cells on ! " << std::endl;

  if(hoRecHits_.size()<nhocells_)
    noisifySubdet(hoRecHits_,hocells_,nhocells_);
  else
    edm::LogWarning("CaloRecHitsProducer") << "All HCAL(HO) cells on ! " << std::endl;

  if(hfRecHits_.size()<nhfcells_)
    noisifySubdet(hfRecHits_,hfcells_,nhfcells_);
  else
    edm::LogWarning("CaloRecHitsProducer") << "All HCAL(HF) cells on ! " << std::endl;
}

void HcalRecHitsMaker::noisifySubdet(std::map<signalHit,float>& theMap, const std::vector<uint32_t>& thecells, unsigned ncells)
{
  unsigned mean=(unsigned)((double)(ncells-theMap.size())*hcalHotFraction_);
  unsigned nhcal = (unsigned)(random_->poissonShoot(mean));
  
  unsigned ncell=0;
  unsigned cellindex=0;
  uint32_t cellnumber=0;
  std::map<signalHit,float>::const_iterator itcheck;

  while(ncell < nhcal)
    {
      cellindex = (unsigned)(random_->flatShoot()*ncells);
      cellnumber = thecells[cellindex];
      itcheck=theMap.find(signalHit(cellnumber));
      if(itcheck==theMap.end()) // new cell
	{
	  theMap.insert(std::pair<signalHit,float>(signalHit(cellnumber),myGaussianTailGenerator_.shoot()));
	  ++ncell;
	}
    }
  //  edm::LogInfo("CaloRecHitsProducer") << "CaloRecHitsProducer : added noise in "<<  ncell << " HCAL cells "  << std::endl;
}

void HcalRecHitsMaker::clean()
{
  hbheRecHits_.clear();
  hfRecHits_.clear();
  hoRecHits_.clear();  
}

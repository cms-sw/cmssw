#include "FastSimulation/CaloRecHitsProducer/interface/HcalRecHitsMaker.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h" 	 
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "CLHEP/GenericFunctions/Erf.hh"

#include "FastSimulation/Utilities/interface/RandomEngine.h"

class RandomEngine;

HcalRecHitsMaker::HcalRecHitsMaker(edm::ParameterSet const & p,
				   const RandomEngine * myrandom)
  :
  initialized_(false),
  random_(myrandom),
  myGaussianTailGenerator_(0)
{
  edm::ParameterSet RecHitsParameters = p.getParameter<edm::ParameterSet>("HCAL");
  noiseHB_ = RecHitsParameters.getParameter<double>("NoiseHB");
  noiseHE_ = RecHitsParameters.getParameter<double>("NoiseHE");
  noiseHO_ = RecHitsParameters.getParameter<double>("NoiseHO");
  noiseHF_ = RecHitsParameters.getParameter<double>("NoiseHF");
  thresholdHB_ = RecHitsParameters.getParameter<double>("ThresholdHB");
  thresholdHE_ = RecHitsParameters.getParameter<double>("ThresholdHE");
  thresholdHO_ = RecHitsParameters.getParameter<double>("ThresholdHO");
  thresholdHF_ = RecHitsParameters.getParameter<double>("ThresholdHF");

  // Computes the fraction of HCAL above the threshold
  Genfun::Erf myErf;

  if(noiseHB_>0.) {
    hcalHotFractionHB_ = 0.5-0.5*myErf(thresholdHB_/noiseHB_/sqrt(2.));
    myGaussianTailGenerator_ = new GaussianTail(random_,noiseHB_,thresholdHB_);
  } else {
    hcalHotFractionHB_ =0.;
  }

  if(noiseHO_>0.) {
    hcalHotFractionHO_ = 0.5-0.5*myErf(thresholdHO_/noiseHO_/sqrt(2.));
    myGaussianTailGenerator_ = new GaussianTail(random_,noiseHO_,thresholdHO_);
  } else {
    hcalHotFractionHO_ =0.;
  }

  if(noiseHE_>0.) {
    hcalHotFractionHE_ = 0.5-0.5*myErf(thresholdHE_/noiseHE_/sqrt(2.));
    myGaussianTailGenerator_ = new GaussianTail(random_,noiseHE_,thresholdHE_);
  } else {
    hcalHotFractionHE_ =0.;
  }

  if(noiseHF_>0.) {
    hcalHotFractionHF_ = 0.5-0.5*myErf(thresholdHF_/noiseHF_/sqrt(2.));
    myGaussianTailGenerator_ = new GaussianTail(random_,noiseHF_,thresholdHF_);
  } else {
    hcalHotFractionHF_ =0.;
  }



}

HcalRecHitsMaker::~HcalRecHitsMaker()
{
  delete myGaussianTailGenerator_;
}

void HcalRecHitsMaker::init(const edm::EventSetup &es)
{
  if(initialized_) return;
  unsigned ncells=createVectorsOfCells(es);
  edm::LogInfo("CaloRecHitsProducer") << "Total number of cells in HCAL " << ncells << std::endl;
  nhbcells_ = hbcells_.size();
  nhecells_ = hecells_.size();
  nhocells_ = hocells_.size();
  nhfcells_ = hfcells_.size(); 
  initialized_=true;
}


// Get the PCaloHits from the event. They have to be stored in a map, because when
// the pile-up is added thanks to the Mixing Module, the same cell can be present several times
void HcalRecHitsMaker::loadPCaloHits(const edm::Event & iEvent)
{

  clean();

  edm::Handle<CrossingFrame> cf;
  iEvent.getByType(cf);
  std::auto_ptr<MixCollection<PCaloHit> > colcalo(new MixCollection<PCaloHit>(cf.product(),"HcalHits",std::pair<int,int>(0,0) ));

  MixCollection<PCaloHit>::iterator it=colcalo->begin();;
  MixCollection<PCaloHit>::iterator itend=colcalo->end();
  unsigned counter=0;
  for(;it!=itend;++it)
    {
      HcalDetId detid(it->id());

      double noise_ = 0.;
      switch(detid.subdet())
	{
	case HcalBarrel: 
	  noise_ = noiseHB_; 
	  Fill(it->id(),it->energy(),hbRecHits_,it.getTrigger(),noise_);
	  break;
	case HcalEndcap: 
	  noise_ = noiseHE_; 
	  Fill(it->id(),it->energy(),heRecHits_,it.getTrigger(),noise_);
	  break;
	case HcalOuter: 
	  noise_ = noiseHO_; 
	  Fill(it->id(),it->energy(),hoRecHits_,it.getTrigger(),noise_);
	  break;		     
	case HcalForward: 
	  noise_ = noiseHF_; 
	  Fill(it->id(),it->energy(),hfRecHits_,it.getTrigger(),noise_);
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

  loadPCaloHits(iEvent);
  noisify();

  // HB
  std::map<uint32_t,std::pair<float,bool> >::const_iterator it=hbRecHits_.begin();
  std::map<uint32_t,std::pair<float,bool> >::const_iterator itend=hbRecHits_.end();
  
  for(;it!=itend;++it)
    {
      // Check if the hit has been killed
      if(it->second.second) continue;
      // Check if it is above the threshold
      if(it->second.first<thresholdHB_) continue; 
      HcalDetId detid(it->first);
      hbheHits.push_back(HBHERecHit(detid,it->second.first,0.));
    }
      
  // HE
  it=heRecHits_.begin();
  itend=heRecHits_.end();  
  for(;it!=itend;++it)
    {
      // Check if the hit has been killed
      if(it->second.second) continue;
      // Check if it is above the threshold
      if(it->second.first<thresholdHE_) continue;
      HcalDetId detid(it->first);
      hbheHits.push_back(HBHERecHit(detid,it->second.first,0.));
    }

  
  // HO
  it = hoRecHits_.begin();
  itend = hoRecHits_.end();
  for(;it!=itend;++it)
    {
      if(it->second.second) continue;
      if(it->second.first<thresholdHE_) continue;
      HcalDetId detid(it->first);
      hoHits.push_back(HORecHit(detid,it->second.first,0));
    }
  
  // HF
  it = hfRecHits_.begin();
  itend = hfRecHits_.end();
  for(;it!=itend;++it)
    {
      if(it->second.second) continue;
      if(it->second.first<thresholdHF_) continue;
      HcalDetId detid(it->first);
      hfHits.push_back(HFRecHit(detid,it->second.first,0));
    }

}


// For a fast injection of the noise: the list of cell ids is stored
unsigned HcalRecHitsMaker::createVectorsOfCells(const edm::EventSetup &es)
{
    edm::ESHandle<CaloGeometry> pG;
    es.get<IdealGeometryRecord>().get(pG);     
    unsigned total=0;
    total += createVectorOfSubdetectorCells(*pG, HcalBarrel,  hbcells_);
    total += createVectorOfSubdetectorCells(*pG, HcalEndcap,  hecells_);
    total += createVectorOfSubdetectorCells(*pG, HcalOuter,   hocells_);
    total += createVectorOfSubdetectorCells(*pG, HcalForward, hfcells_);    
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

// Takes a hit (from a PSimHit) and fills a map 
void HcalRecHitsMaker::Fill(uint32_t id, float energy, std::map<uint32_t,std::pair<float,bool> >& myHits, bool signal, double noise_)
{

  // The signal hits are singletons (no need to look into a map)
  if(signal)
    {
      // a new hit is created
      // we can give a hint for the insert
      // Add the noise at this point. We are sure that it won't be added several times

      if(noise_ > 0.) energy += random_->gaussShoot(0.,noise_);
      std::pair<float,bool> hit(energy,false); 
      // if it is signal, it is already ordered, so we can give a hint for the 
      // insert
      if(signal)
	myHits.insert(myHits.end(),std::pair<uint32_t,std::pair<float,bool> >(id,hit));
    }
  else       // In this case,there is a risk of duplication. Need to look into the map
    {
      std::map<uint32_t,std::pair<float,bool> >::iterator itcheck=myHits.find(id);
      if(itcheck==myHits.end())
	{
	  if(noise_ > 0.) energy += random_->gaussShoot(0.,noise_);
	  std::pair<float,bool> hit(energy,false); 
	  myHits.insert(std::pair<uint32_t,std::pair<float,bool> >(id,hit));
	}
      else
	{
	  itcheck->second.first += energy;
	}
    }
}

void HcalRecHitsMaker::noisify()
{

  if(noiseHB_ > 0.) {
    if(hbRecHits_.size()<nhbcells_)
      {
	// No need to do it anymore. The noise on the signal has been added 
	// when loading the PCaloHits
	// noisifySignal(hbheRecHits_);      
	noisifySubdet(hbRecHits_,hbcells_,nhbcells_,hcalHotFractionHB_);
      }
    else
      edm::LogWarning("CaloRecHitsProducer") << "All HCAL (HB) cells on ! " << std::endl;
  }


  if(noiseHE_ > 0.) {
    if(heRecHits_.size()<nhecells_)
      {
	// No need to do it anymore. The noise on the signal has been added 
	// when loading the PCaloHits
	// noisifySignal(hbheRecHits_);      
	noisifySubdet(heRecHits_,hecells_,nhecells_,hcalHotFractionHE_);
      }
    else
      edm::LogWarning("CaloRecHitsProducer") << "All HCAL (HE) cells on ! " << std::endl;
  }

  if(noiseHO_ > 0.) {
    if( hoRecHits_.size()<nhocells_)
      {
	//      noisifySignal(hoRecHits_);
	noisifySubdet(hoRecHits_,hocells_,nhocells_,hcalHotFractionHO_);
      }
    else
      edm::LogWarning("CaloRecHitsProducer") << "All HCAL(HO) cells on ! " << std::endl;
  }
   
  if(noiseHF_ > 0.) {
    if(hfRecHits_.size()<nhfcells_)
      {
	//      noisifySignal(hfRecHits_);
	noisifySubdet(hfRecHits_,hfcells_,nhfcells_,hcalHotFractionHF_);
      }
    else
      edm::LogWarning("CaloRecHitsProducer") << "All HCAL(HF) cells on ! " << std::endl;
  }
}

void HcalRecHitsMaker::noisifySubdet(std::map<uint32_t,std::pair<float,bool> >& theMap, const std::vector<uint32_t>& thecells, unsigned ncells, double hcalHotFraction_)
{

  unsigned mean=(unsigned)((double)(ncells-theMap.size())*hcalHotFraction_);
  unsigned nhcal = (unsigned)(random_->poissonShoot(mean));
  
  unsigned ncell=0;
  unsigned cellindex=0;
  uint32_t cellnumber=0;
  std::map<uint32_t,std::pair<float,bool> >::const_iterator itcheck;

  while(ncell < nhcal)
    {
      cellindex = (unsigned)(random_->flatShoot()*ncells);
      cellnumber = thecells[cellindex];
      itcheck=theMap.find(cellnumber);
      if(itcheck==theMap.end()) // new cell
	{
	  std::pair <float,bool> noisehit(myGaussianTailGenerator_->shoot(),false);
	  theMap.insert(std::pair<uint32_t,std::pair<float,bool> >(cellnumber,noisehit));
	  ++ncell;
	}
    }
   edm::LogInfo("CaloRecHitsProducer") << "CaloRecHitsProducer : added noise in "<<  ncell << " HCAL cells "  << std::endl;
}

void HcalRecHitsMaker::clean()
{
  hbRecHits_.clear();
  heRecHits_.clear();
  hfRecHits_.clear();
  hoRecHits_.clear();  
}

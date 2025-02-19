#include "FastSimulation/CaloRecHitsProducer/interface/EcalPreshowerRecHitsMaker.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h" 	 
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "FastSimulation/Utilities/interface/RandomEngine.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "CLHEP/GenericFunctions/Erf.hh"

EcalPreshowerRecHitsMaker::EcalPreshowerRecHitsMaker(
  edm::ParameterSet const & p, 
  const RandomEngine * myrandom)
  :
  random_(myrandom),
  myGaussianTailGenerator_(0)
{
  edm::ParameterSet RecHitsParameters 
    = p.getParameter<edm::ParameterSet>("ECALPreshower");
  noise_ = RecHitsParameters.getParameter<double>("Noise");
  threshold_ = RecHitsParameters.getParameter<double>("Threshold");
  inputCol_=RecHitsParameters.getParameter<edm::InputTag>("MixedSimHits");

  initialized_=false;

  Genfun::Erf myErf;
  if(  noise_>0. ) {
    preshowerHotFraction_ = 0.5-0.5*myErf(threshold_/noise_/sqrt(2.));
    myGaussianTailGenerator_ = new GaussianTail(random_, noise_, threshold_);
  } else {
    preshowerHotFraction_ =0.;
  }
}

EcalPreshowerRecHitsMaker::~EcalPreshowerRecHitsMaker()
{
  initialized_=false;
  delete myGaussianTailGenerator_;
}

void EcalPreshowerRecHitsMaker::clean()
{

  ecalsRecHits_.clear();
}


void EcalPreshowerRecHitsMaker::loadEcalPreshowerRecHits(edm::Event &iEvent,ESRecHitCollection & ecalHits)
{
  // if no preshower, do nothing

  if(ncells_==0) return;
  loadPCaloHits(iEvent);
  if( myGaussianTailGenerator_ ) noisify();

  std::map<uint32_t,std::pair<float,bool> >::const_iterator 
    it=ecalsRecHits_.begin();
  std::map<uint32_t,std::pair<float,bool> >::const_iterator 
    itend=ecalsRecHits_.end();

  for(;it!=itend;++it)
    {
      // check if the hit has been killed 
      if(it->second.second) continue;
      // check if it is above the threshold
      if(it->second.first<threshold_) continue;
      ESDetId detid(it->first);
      //  std::cout << detid << " " << it->second.first << std::endl;
      ecalHits.push_back(EcalRecHit(detid,it->second.first,0.)); 
    }
}

void EcalPreshowerRecHitsMaker::loadPCaloHits(const edm::Event & iEvent)
{

  clean();

  edm::Handle<CrossingFrame<PCaloHit> > cf;
  iEvent.getByLabel(inputCol_,cf);
  std::auto_ptr<MixCollection<PCaloHit> > colcalo(new MixCollection<PCaloHit>(cf.product(),std::pair<int,int>(0,0) ));

  MixCollection<PCaloHit>::iterator it=colcalo->begin();
  MixCollection<PCaloHit>::iterator itend=colcalo->end();
  for(;it!=itend;++it)
    {
      Fill(it->id(),it->energy(),ecalsRecHits_,it.getTrigger());
      //      Fill(it->id(),it->energy(),ecalsRecHits_);
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
    es.get<CaloGeometryRecord>().get(pG);     
    unsigned total=0;

    escells_.reserve(137728);

    const CaloSubdetectorGeometry* geom=pG->getSubdetectorGeometry(DetId::Ecal,EcalPreshower);  
    if(geom==0) return 0;
    const std::vector<DetId>& ids(geom->getValidDetIds(DetId::Ecal,EcalPreshower));  
    for (std::vector<DetId>::const_iterator i=ids.begin(); i!=ids.end(); i++) 
      {
	escells_.push_back(i->rawId());
	++total;
      }
    return escells_.size();
}

void EcalPreshowerRecHitsMaker::noisify()
{
  if(ecalsRecHits_.size()<ncells_) 
    {
      // Not needed anymore, the noise is added when loading the PCaloHits
      // noisifySignal(ecalsRecHits_);
      noisifySubdet(ecalsRecHits_,escells_,ncells_);
    }
  else
    edm::LogWarning("CaloRecHitsProducer") << "All Preshower cells on ! " << std::endl;
}


void EcalPreshowerRecHitsMaker::noisifySubdet(std::map<uint32_t,std::pair<float,bool> >& theMap, const std::vector<uint32_t>& thecells, unsigned ncells)
{
  // noise won't be injected in cells that contain signal
  double mean = (double)(ncells-theMap.size())*preshowerHotFraction_;
  unsigned nps = random_->poissonShoot(mean);
  
  unsigned ncell=0;
  unsigned cellindex=0;
  uint32_t cellnumber=0;
  std::map<uint32_t,std::pair<float,bool> >::const_iterator itcheck;

  while(ncell < nps)
    {
      cellindex = (unsigned)(random_->flatShoot()*ncells);
      cellnumber = thecells[cellindex];
      itcheck=theMap.find(cellnumber);
      if(itcheck==theMap.end()) // inject only in empty cells
	{
	  std::pair <float,bool> noisehit(myGaussianTailGenerator_->shoot(),
					  false);
	  theMap.insert(std::pair<uint32_t,std::pair<float,bool> >
			(cellnumber,noisehit));
	  ++ncell;
	}
    }
  //  edm::LogInfo("CaloRecHitsProducer") << "CaloRecHitsProducer : added noise in "<<  ncell << " HCAL cells "  << std::endl;
}


// Takes a hit (from a PSimHit) and fills a map 
void EcalPreshowerRecHitsMaker::Fill(uint32_t id,float energy, std::map<uint32_t,std::pair<float,bool> >& myHits,bool signal)
{
  // The signal hits are singletons (no need to look into a map)
  if(signal)
    {
      // a new hit is created
      // we can give a hint for the insert
      // Add the noise at this point. We are sure that it won't be added several times
      energy += random_->gaussShoot(0.,noise_);
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
	  energy += random_->gaussShoot(0.,noise_);
	  std::pair<float,bool> hit(energy,false); 	  
	  myHits.insert(std::pair<uint32_t,std::pair<float,bool> >(id,hit));
	}
      else
	{
	  itcheck->second.first += energy;
	}
    }  
}
/*
void EcalPreshowerRecHitsMaker::noisifySignal(std::map<uint32_t,std::pair<float,bool> >& theMap)
{
  std::map<uint32_t,std::pair<float,bool> >::iterator it=theMap.begin();
  std::map<uint32_t,std::pair<float,bool> >::iterator itend=theMap.end();
  for(;it!=itend;++it)
    {
      it->second.first+= random_->gaussShoot(0.,noise_);
      if(it->second.first < threshold_)
	{
	  it->second.second=true;
	  it->second.first = 0.;
	}
    }
}
*/

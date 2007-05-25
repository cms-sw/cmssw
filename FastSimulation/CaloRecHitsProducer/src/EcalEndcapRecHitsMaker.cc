#include "FastSimulation/CaloRecHitsProducer/interface/EcalEndcapRecHitsMaker.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "FastSimulation/Utilities/interface/RandomEngine.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h" 	 
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/EcalEndcapAlgo/interface/EcalEndcapGeometry.h"



EcalEndcapRecHitsMaker::EcalEndcapRecHitsMaker(edm::ParameterSet const & p, 
					       const RandomEngine * myrandom) 
  : random_(myrandom)
{
  edm::ParameterSet RecHitsParameters = p.getParameter<edm::ParameterSet>("ECALEndcap");
  noise_ = RecHitsParameters.getParameter<double>("Noise");
  threshold_ = RecHitsParameters.getParameter<double>("Threshold");
  theCalorimeterHits_.resize(20000,0.);
  noisified_ = (noise_==0.);
}

EcalEndcapRecHitsMaker::~EcalEndcapRecHitsMaker()
{;
}

void EcalEndcapRecHitsMaker::clean()
{

  ecaleRecHits_.clear();

  unsigned size=theFiredCells_.size();
  for(unsigned ic=0;ic<size;++ic)
    {
      theCalorimeterHits_[theFiredCells_[ic]] = 0.;
    }
  theFiredCells_.clear();
  // If the noise is set to 0. No need to simulate it. 
  noisified_ = (noise_==0.);
}


void EcalEndcapRecHitsMaker::loadEcalEndcapRecHits(edm::Event &iEvent,EERecHitCollection & ecalHits)
{


  clean();

  loadPCaloHits(iEvent);

  unsigned nhit=theFiredCells_.size();

  for(unsigned ihit=0;ihit<nhit;++ihit)
    {      
      unsigned icell = theFiredCells_[ihit];
      // It is safer to update the orignal array in case this methods is called several times
      if (!noisified_ )  theCalorimeterHits_[icell] += random_->gaussShoot(0.,noise_);
      
      // If the energy+noise is below the threshold, a hit is nevertheless created, otherwise, there is a risk that a "noisy" hit 
      // is afterwards put in this cell which would not be correct. 
      if (  theCalorimeterHits_[icell]<threshold_ ) theCalorimeterHits_[icell]=0.;
      ecalHits.push_back(EcalRecHit(EEDetId(endcapRawId_[icell]),theCalorimeterHits_[icell],0.));
    }
  noisified_ = true;

}

void EcalEndcapRecHitsMaker::loadPCaloHits(const edm::Event & iEvent)
{

  edm::Handle<CrossingFrame> cf;
  iEvent.getByType(cf);
  std::auto_ptr<MixCollection<PCaloHit> > colcalo(new MixCollection<PCaloHit>(cf.product(),"EcalHitsEE",std::pair<int,int>(0,0) ));

  theFiredCells_.reserve(colcalo->size());

  MixCollection<PCaloHit>::iterator cficalo;
  MixCollection<PCaloHit>::iterator cficaloend=colcalo->end();

  for (cficalo=colcalo->begin(); cficalo!=cficaloend;cficalo++) 
    {
      unsigned hashedindex = EEDetId(cficalo->id()).hashedIndex();      
      // Check if the hit already exists
      if(theCalorimeterHits_[hashedindex]==0.)
	{
	  theFiredCells_.push_back(hashedindex); 
	}
      theCalorimeterHits_[hashedindex]+=cficalo->energy();   
    }
}

// Takes a hit (from a PSimHit) and fills a map with it after adding the noise. 
void EcalEndcapRecHitsMaker::noisifyAndFill(uint32_t id,float energy, std::map<uint32_t,float>& myHits)
{


  if (noise_>0.) energy += random_->gaussShoot(0.,noise_);

  // If below the threshold, a hit is nevertheless created, otherwise, there is a risk that a "noisy" hit 
  // is afterwards put in this cell which would not be correct. 
  if ( energy <threshold_ ) energy=0.;
  // No double counting check. Depending on how the pile-up is implemented , this can be a problem.
  // Fills the map giving a "hint", in principle, the objets have already been ordered in CalorimetryManager
  myHits.insert(myHits.end(),std::pair<uint32_t,float>(id,energy));
}

void EcalEndcapRecHitsMaker::init(const edm::EventSetup &es)
{
  endcapRawId_.resize(20000);
  edm::ESHandle<CaloGeometry> pG;
  es.get<IdealGeometryRecord>().get(pG);   
  
  const EcalEndcapGeometry * myEcalEndcapGeometry = dynamic_cast<const EcalEndcapGeometry*>(pG->getSubdetectorGeometry(DetId::Ecal,EcalEndcap));
  std::vector<DetId> vec(myEcalEndcapGeometry->getValidDetIds(DetId::Ecal,EcalEndcap));
  unsigned size=vec.size();    
  for(unsigned ic=0; ic<size; ++ic) 
    {
      endcapRawId_[EEDetId(vec[ic]).hashedIndex()]=vec[ic].rawId();
    }
  //  std::cout << " Made the array " << std::endl;
}

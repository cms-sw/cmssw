#include "FastSimulation/CaloRecHitsProducer/interface/EcalBarrelRecHitsMaker.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "FastSimulation/Utilities/interface/RandomEngine.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h" 	 
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/EcalBarrelAlgo/interface/EcalBarrelGeometry.h"

EcalBarrelRecHitsMaker::EcalBarrelRecHitsMaker(edm::ParameterSet const & p,
					       edm::ParameterSet const & pcalib,
					       const RandomEngine* myrandom)
  : random_(myrandom)
{
  edm::ParameterSet RecHitsParameters = p.getParameter<edm::ParameterSet>("ECALBarrel");
  noise_ = RecHitsParameters.getParameter<double>("Noise");
  threshold_ = RecHitsParameters.getParameter<double>("Threshold");
  theCalorimeterHits_.resize(62000,0.);
  noisified_ = (noise_==0.);
  double c1=pcalib.getParameter<double>("EEs25notContainment"); 
  calibfactor_=1./c1;
}

EcalBarrelRecHitsMaker::~EcalBarrelRecHitsMaker()
{;
}


void EcalBarrelRecHitsMaker::clean()
{
  //  std::cout << " clean " << std::endl;
  ecalbRecHits_.clear();
  unsigned size=theFiredCells_.size();
  for(unsigned ic=0;ic<size;++ic)
    {
      theCalorimeterHits_[theFiredCells_[ic]] = 0.;
    }
  theFiredCells_.clear();
  // If the noise is set to 0. No need to simulate it. 
  noisified_ = (noise_==0.);
  //  std::cout << " Finished to clean "  << std::endl;
}

void EcalBarrelRecHitsMaker::loadEcalBarrelRecHits(edm::Event &iEvent,EBRecHitCollection & ecalHits)
{

  clean();
  loadPCaloHits(iEvent);
  
  unsigned nhit=theFiredCells_.size();
  //  std::cout << " loadEcalBarrelRecHits " << nhit << std::endl;

  for(unsigned ihit=0;ihit<nhit;++ihit)
    {      
      unsigned icell = theFiredCells_[ihit];
      //      std::cout << " The Fired Cell " << icell << std::endl;
      // It is safer to update the orignal array in case this methods is called several times
      if (!noisified_ )  theCalorimeterHits_[icell] += random_->gaussShoot(0.,noise_);
      
      //      std::cout << "Noise ok " << std::endl;

      // If the energy+noise is below the threshold, a hit is nevertheless created, otherwise, there is a risk that a "noisy" hit 
      // is afterwards put in this cell which would not be correct. 
      if (  theCalorimeterHits_[icell]<threshold_ ) theCalorimeterHits_[icell]=0.;
//      std::cout << " Threshold ok " << std::endl;
//      std::cout << " Raw Id " << barrelRawId_[icell] << std::endl;
      ecalHits.push_back(EcalRecHit(EBDetId(barrelRawId_[icell]),theCalorimeterHits_[icell],0.));
      //      std::cout << " Hit stored " << std::endl;
    }
  //  std::cout << " Done " << std::endl;
  noisified_ = true;
}

void EcalBarrelRecHitsMaker::loadPCaloHits(const edm::Event & iEvent)
{

  edm::Handle<CrossingFrame> cf;
  iEvent.getByType(cf);
  std::auto_ptr<MixCollection<PCaloHit> > colcalo(new MixCollection<PCaloHit>(cf.product(),"EcalHitsEB",std::pair<int,int>(0,0) ));


  theFiredCells_.reserve(colcalo->size());

  MixCollection<PCaloHit>::iterator cficalo;
  MixCollection<PCaloHit>::iterator cficaloend=colcalo->end();

  for (cficalo=colcalo->begin(); cficalo!=cficaloend;cficalo++) 
    {
      unsigned hashedindex = EBDetId(cficalo->id()).hashedIndex();      
      // Check if the hit already exists
      if(theCalorimeterHits_[hashedindex]==0.)
	{
	  theFiredCells_.push_back(hashedindex); 

	}
      // the famous 1/0.97 calibration factor is applied here ! 
      theCalorimeterHits_[hashedindex]+=cficalo->energy()*calibfactor_;   
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

void EcalBarrelRecHitsMaker::init(const edm::EventSetup &es)
{
  barrelRawId_.resize(62000);
  edm::ESHandle<CaloGeometry> pG;
  es.get<IdealGeometryRecord>().get(pG);   
  
  const EcalBarrelGeometry * myEcalBarrelGeometry = dynamic_cast<const EcalBarrelGeometry*>(pG->getSubdetectorGeometry(DetId::Ecal,EcalBarrel));
  //  std::cout << " Got the geometry " << myEcalBarrelGeometry << std::endl;
  std::vector<DetId> vec(myEcalBarrelGeometry->getValidDetIds(DetId::Ecal,EcalBarrel));
  unsigned size=vec.size();    
  for(unsigned ic=0; ic<size; ++ic) 
    {
      barrelRawId_[EBDetId(vec[ic]).hashedIndex()]=vec[ic].rawId();
    }
  //  std::cout << " Made the array " << std::endl;
}

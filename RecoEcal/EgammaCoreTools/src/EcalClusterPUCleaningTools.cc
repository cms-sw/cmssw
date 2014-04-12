#include "RecoEcal/EgammaCoreTools/interface/EcalClusterPUCleaningTools.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"

#include "RecoEcal/EgammaCoreTools/interface/SuperClusterShapeAlgo.h"

EcalClusterPUCleaningTools::EcalClusterPUCleaningTools( const edm::Event &ev, const edm::EventSetup &es, const edm::InputTag& redEBRecHits, const edm::InputTag& redEERecHits )
{
  getGeometry( es );
  getEBRecHits( ev, redEBRecHits );
  getEERecHits( ev, redEERecHits );
}



EcalClusterPUCleaningTools::~EcalClusterPUCleaningTools()
{
}


void EcalClusterPUCleaningTools::getGeometry( const edm::EventSetup &es )
{
  edm::ESHandle<CaloGeometry> pGeometry;
  es.get<CaloGeometryRecord>().get(pGeometry);
  geometry_ = pGeometry.product();
}

void EcalClusterPUCleaningTools::getEBRecHits( const edm::Event &ev, const edm::InputTag& redEBRecHits )
{
  edm::Handle< EcalRecHitCollection > pEBRecHits;
  ev.getByLabel( redEBRecHits, pEBRecHits );
  ebRecHits_ = pEBRecHits.product();
}



void EcalClusterPUCleaningTools::getEERecHits( const edm::Event &ev, const edm::InputTag& redEERecHits )
{
  edm::Handle< EcalRecHitCollection > pEERecHits;
  ev.getByLabel( redEERecHits, pEERecHits );
  eeRecHits_ = pEERecHits.product();
}


reco::SuperCluster EcalClusterPUCleaningTools::CleanedSuperCluster(float xi, const reco::SuperCluster &scluster, const edm::Event &ev ){
  
  //std::cout << "\nEcalClusterPUCleaningTools::CleanedSuperCluster called, this will give you back a cleaned supercluster" << std::endl;

  // seed basic cluster of initial SC: this will remain in the cleaned SC, by construction
  reco::CaloClusterPtr seed = scluster.seed();

  float seedBCEnergy      = (scluster.seed())->energy(); // this should be replaced by the 5x5 around the seed; a good approx of how E_seed is defined 
  float eSeed             = 0.35;                        // standard eSeed in EB ; see CMS IN-2010/008
  int   numBcRemoved      = 0;  

  double ClusterE = 0; //Sum of cluster energies for supercluster.
  //Holders for position of this supercluster.
  double posX=0;
  double posY=0;
  double posZ=0;

  reco::CaloClusterPtrVector thissc;

  // looping on basic clusters within the Supercluster
  for(reco::CaloCluster_iterator bcIt = scluster.clustersBegin(); bcIt!=scluster.clustersEnd(); bcIt++)
    {
      // E_seed is an Et  selection on 5x1 dominoes (around which BC's are built), see CMS IN-2010/008
      // here such selection is implemented on the full BC around it 
      if( (*bcIt)->energy() > sqrt( eSeed*eSeed + xi*xi*seedBCEnergy*seedBCEnergy/cosh((*bcIt)->eta())/cosh((*bcIt)->eta())  ) ) 
	{
	  ;
	}// the sum only of the BC's that pass the Esee selection
      else{
	numBcRemoved++;
	continue;
      }// count how many basic cluster get removed by the cleaning
      
      // if BC passes dynamic selection, include it in the 'cleaned' supercluster
      thissc.push_back( (*bcIt) );
      // cumulate energy and position of the cleaned supercluster
      ClusterE += (*bcIt)->energy();
      posX += (*bcIt)->energy() * (*bcIt)->position().X();
      posY += (*bcIt)->energy() * (*bcIt)->position().Y();
      posZ += (*bcIt)->energy() * (*bcIt)->position().Z();
      
    }// loop on basic clusters of the original SC

  posX /= ClusterE;
  posY /= ClusterE;
  posZ /= ClusterE;

  // make temporary 'cleaned' supercluster in oder to compute the covariances 
  double Epreshower=scluster.preshowerEnergy();
  double phiWidth=0.; 
  double etaWidth=0.; 
  reco::SuperCluster suCltmp(ClusterE, math::XYZPoint(posX, posY, posZ), seed, thissc, Epreshower, phiWidth, etaWidth);
  

  // construct cluster shape to compute ieta and iphi covariances of the SC 
  const CaloSubdetectorGeometry *geometry_p=0;
  if (seed->seed().det() == DetId::Ecal && seed->seed().subdetId() == EcalBarrel){
    geometry_p = geometry_->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
    SuperClusterShapeAlgo  SCShape(   ebRecHits_ , geometry_p); 
    SCShape.Calculate_Covariances( suCltmp );
    phiWidth= SCShape.phiWidth(); 
    etaWidth= SCShape.etaWidth(); 
  }
  else if (seed->seed().det() == DetId::Ecal && seed->seed().subdetId() == EcalEndcap){
    geometry_p = geometry_->getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
    SuperClusterShapeAlgo  SCShape(   eeRecHits_ , geometry_p); 
    SCShape.Calculate_Covariances( suCltmp );
    phiWidth= SCShape.phiWidth(); 
    etaWidth= SCShape.etaWidth(); 
  }
  else {
    std::cout << "The seed crystal of this SC is neither in EB nor in EE. This is a problem. Bailing out " << std::endl;
    assert(-1);
  }
  
  // return the cleaned supercluster SCluster, with covariances updated 
  return reco::SuperCluster(ClusterE, math::XYZPoint(posX, posY, posZ), seed, thissc, Epreshower, phiWidth, etaWidth);

}

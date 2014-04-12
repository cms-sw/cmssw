#include "RecoParticleFlow/PFClusterTools/interface/PFClusterWidthAlgo.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h" 
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "TMath.h"
#include <algorithm>
using namespace std;
using namespace reco;



PFClusterWidthAlgo::PFClusterWidthAlgo(const std::vector<const reco::PFCluster *>& pfclust) {


  double numeratorEtaWidth = 0.;
  double numeratorPhiWidth = 0.;
  double sclusterE = 0.;
  double posX = 0.;
  double posY = 0.;
  double posZ = 0.;
  sigmaEtaEta_ = 0.;

  unsigned int nclust= pfclust.size();
  if(nclust == 0 ) {
    etaWidth_ = 0.;
    phiWidth_ = 0.;
    sigmaEtaEta_ = 0.;
  }
  else {
    
    //first loop, compute supercluster position at ecal face, and energy sum from rechit loop
    //in order to be consistent with variance calculation
    for(unsigned int icl=0; icl<nclust; ++icl) {
      const std::vector< reco::PFRecHitFraction >& PFRecHits =  pfclust[icl]->recHitFractions();
      
      for ( std::vector< reco::PFRecHitFraction >::const_iterator it = PFRecHits.begin(); 
            it != PFRecHits.end(); ++it) {
        const PFRecHitRef& RefPFRecHit = it->recHitRef(); 
        //compute rechit energy taking into account fractions
        double energyHit = RefPFRecHit->energy()*it->fraction();

        sclusterE += energyHit;
        posX += energyHit*RefPFRecHit->position().X();
        posY += energyHit*RefPFRecHit->position().Y();
        posZ += energyHit*RefPFRecHit->position().Z();
      
      }
    } // end for ncluster    

    double denominator = sclusterE;
    
    posX /=sclusterE;
    posY /=sclusterE;
    posZ /=sclusterE;

    math::XYZPoint pflowSCPos(posX,posY,posZ);

    double scEta    = pflowSCPos.eta();
    double scPhi    = pflowSCPos.phi();    
    
    double SeedClusEnergy = -1.;
    unsigned int SeedDetID = 0;
    double SeedEta = -1.;

    //second loop, compute variances
    for(unsigned int icl=0; icl<nclust; ++icl) {
      const std::vector< reco::PFRecHitFraction >& PFRecHits =  pfclust[icl]->recHitFractions();  
      
      for ( std::vector< reco::PFRecHitFraction >::const_iterator it = PFRecHits.begin(); 
	    it != PFRecHits.end(); ++it) {
	const PFRecHitRef& RefPFRecHit = it->recHitRef(); 
        //compute rechit energy taking into account fractions
	double energyHit = RefPFRecHit->energy()*it->fraction();

	//only for the first cluster (from GSF) find the seed
	if(icl==0) {
	  if (energyHit > SeedClusEnergy) {
	    SeedClusEnergy = energyHit;
	    SeedEta = RefPFRecHit->position().eta();
	    SeedDetID = RefPFRecHit->detId();
	  }
	}

	double dPhi = reco::deltaPhi(RefPFRecHit->position().phi(),scPhi);
	double dEta = RefPFRecHit->position().eta() - scEta;
	numeratorEtaWidth += energyHit * dEta * dEta;
	numeratorPhiWidth += energyHit * dPhi * dPhi;
      }
    } // end for ncluster

    //for the first cluster (from GSF) computed sigmaEtaEta
    const std::vector< reco::PFRecHitFraction >& PFRecHits =  pfclust[0]->recHitFractions();
    for ( std::vector< reco::PFRecHitFraction >::const_iterator it = PFRecHits.begin(); 
	  it != PFRecHits.end(); ++it) {
      const PFRecHitRef& RefPFRecHit = it->recHitRef(); 
      if(!RefPFRecHit.isAvailable()) 
	return;
      double energyHit = RefPFRecHit->energy();
      if (RefPFRecHit->detId() != SeedDetID) {
	float diffEta =  RefPFRecHit->position().eta() - SeedEta;
	sigmaEtaEta_ += (diffEta*diffEta) * (energyHit/SeedClusEnergy);
      }
    }
    if (sigmaEtaEta_ == 0.) sigmaEtaEta_ = 0.00000001;

    etaWidth_ = sqrt(numeratorEtaWidth / denominator);
    phiWidth_ = sqrt(numeratorPhiWidth / denominator);
    

  } // endif ncluster > 0
}
PFClusterWidthAlgo::~PFClusterWidthAlgo()
{
}

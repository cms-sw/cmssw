#include "RecoParticleFlow/PFClusterTools/interface/PFClusterWidthAlgo.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h" 
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "TMath.h"
#include <algorithm>
using namespace std;
using namespace reco;



PFClusterWidthAlgo::PFClusterWidthAlgo(const std::vector<const reco::PFCluster *>& pfclust,
				       const EBRecHitCollection * ebRecHits,
				       const EERecHitCollection * eeRecHits){


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

    for(unsigned int icl=0;icl<nclust;++icl) {
      double e = pfclust[icl]->energy();
      sclusterE += e;
      posX += e * pfclust[icl]->position().X();
      posY += e * pfclust[icl]->position().Y();
      posZ += e * pfclust[icl]->position().Z();	  
    }
    
    posX /=sclusterE;
    posY /=sclusterE;
    posZ /=sclusterE;
    
    double denominator = sclusterE;
    
    math::XYZPoint pflowSCPos(posX,posY,posZ);
    
    double scEta    = pflowSCPos.eta();
    double scPhi    = pflowSCPos.phi();
    
    double SeedClusEnergy = -1.;
    unsigned int SeedDetID = 0;
    double SeedEta = -1.;
    double SeedPhi = -1.;

    for(unsigned int icl=0; icl<nclust; ++icl) {
      const std::vector< reco::PFRecHitFraction >& PFRecHits =  pfclust[icl]->recHitFractions();
      
      
      for ( std::vector< reco::PFRecHitFraction >::const_iterator it = PFRecHits.begin(); 
	    it != PFRecHits.end(); ++it) {
	const PFRecHitRef& RefPFRecHit = it->recHitRef(); 
	double energyHit=0;
	// if the PFRecHit is not available, try to get it from the collections
	//	if(!RefPFRecHit.isAvailable() && ebRecHits && eeRecHits )
	if(ebRecHits && eeRecHits )
	  {
	    //	    std::cout << " Recomputing " << std::endl;
	    unsigned index=it-PFRecHits.begin();
	    DetId id=pfclust[icl]->hitsAndFractions()[index].first;
	    // look for the hit; do not forget to multiply by the fraction
	    if(id.det()==DetId::Ecal && id.subdetId()==EcalBarrel)
	      {
		EBRecHitCollection::const_iterator itcheck=ebRecHits->find(id);
		if(itcheck!=ebRecHits->end())
		  energyHit= itcheck->energy();
		// The cluster shapes do not take into account the RecHit fraction !   		
		// * pfclust[icl]->hitsAndFractions()[index].second;
	      }
	    if(id.det()==DetId::Ecal && id.subdetId()==EcalEndcap)
	      {
		EERecHitCollection::const_iterator itcheck=eeRecHits->find(id);
		if(itcheck!=eeRecHits->end())
		  energyHit= itcheck->energy();
		// The cluster shapes do not take into account the RecHit fraction !   		
		//               * pfclust[icl]->hitsAndFractions()[index].second;
	      }
	  }
	else
	  energyHit = RefPFRecHit->energy();

	//only for the first cluster (from GSF) find the seed
	if(icl==0) {
	  if (energyHit > SeedClusEnergy) {
	    SeedClusEnergy = energyHit;
	    SeedEta = RefPFRecHit->position().eta();
	    SeedPhi =  RefPFRecHit->position().phi();
	    SeedDetID = RefPFRecHit->detId();
	  }
	}


	double dPhi = RefPFRecHit->position().phi() - scPhi;
	if (dPhi > + TMath::Pi()) { dPhi = TMath::TwoPi() - dPhi; }
	if (dPhi < - TMath::Pi()) { dPhi = TMath::TwoPi() + dPhi; }
	double dEta = RefPFRecHit->position().eta() - scEta;
	if ( energyHit > 0 ) {
	  numeratorEtaWidth += energyHit * dEta * dEta;
	  numeratorPhiWidth += energyHit * dPhi * dPhi;
	}
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

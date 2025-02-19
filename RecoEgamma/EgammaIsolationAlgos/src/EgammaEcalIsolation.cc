//*****************************************************************************
// File:      EgammaEcalIsolation.cc
// ----------------------------------------------------------------------------
// OrigAuth:  Gilles De Lentdecker
// Institute: IIHE-ULB
//=============================================================================
//*****************************************************************************

//C++ includes
#include <vector>
#include <functional>

//ROOT includes
#include <Math/VectorUtil.h>

//CMSSW includes
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaEcalIsolation.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

using namespace ROOT::Math::VectorUtil ;


EgammaEcalIsolation::EgammaEcalIsolation (double extRadius,
					  double etLow,
					  const reco::BasicClusterCollection* basicClusterCollection,
					  const reco::SuperClusterCollection* superClusterCollection):
  etMin(etLow),
  conesize(extRadius),
  basicClusterCollection_(basicClusterCollection),
  superClusterCollection_(superClusterCollection)
{
}

EgammaEcalIsolation::~EgammaEcalIsolation(){}

double EgammaEcalIsolation::getEcalEtSum(const reco::Candidate* candidate){

  
  double ecalIsol=0.;
  reco::SuperClusterRef sc = candidate->get<reco::SuperClusterRef>();
  math::XYZVector position(sc.get()->position().x(),
		   sc.get()->position().y(),
		   sc.get()->position().z());
  
  // match the photon hybrid supercluster with those with Algo==0 (island)
  double delta1=1000.;
  double deltacur=1000.;
  const reco::SuperCluster *matchedsupercluster=0;
  bool MATCHEDSC = false;
  
  for(reco::SuperClusterCollection::const_iterator scItr = superClusterCollection_->begin(); scItr != superClusterCollection_->end(); ++scItr){
    
    const reco::SuperCluster *supercluster = &(*scItr);
 
    math::XYZVector currentPosition(supercluster->position().x(),
		     supercluster->position().y(),
		     supercluster->position().z());
 
    
    if(supercluster->seed()->algo() == 0){
      deltacur = DeltaR(currentPosition, position); 
      
      if (deltacur < delta1) {
        delta1=deltacur;
	matchedsupercluster = supercluster;
	MATCHEDSC = true;
      }
    }
  }

  const reco::BasicCluster *cluster= 0;
  
  //loop over basic clusters
  for(reco::BasicClusterCollection::const_iterator cItr = basicClusterCollection_->begin(); cItr != basicClusterCollection_->end(); ++cItr){
 
    cluster = &(*cItr);
//    double ebc_bcchi2 = cluster->chi2();
    int   ebc_bcalgo = cluster->algo();
    double ebc_bce    = cluster->energy();
    double ebc_bceta  = cluster->eta();
    double ebc_bcet   = ebc_bce*sin(2*atan(exp(ebc_bceta)));
    double newDelta = 0.;


    if (ebc_bcet > etMin && ebc_bcalgo == 0) {
  //    if (ebc_bcchi2 < 30.) {
	
	if(MATCHEDSC){
	  bool inSuperCluster = false;

	  reco::CaloCluster_iterator theEclust = matchedsupercluster->clustersBegin();
	  // loop over the basic clusters of the matched supercluster
	  for(;theEclust != matchedsupercluster->clustersEnd();
	      theEclust++) {
	    if ((**theEclust) ==  (*cluster) ) inSuperCluster = true;
	  }
	  if (!inSuperCluster) {

	    math::XYZVector basicClusterPosition(cluster->position().x(),
						  cluster->position().y(),
						  cluster->position().z());
	    newDelta=DeltaR(basicClusterPosition,position);
	    if(newDelta < conesize) {
	      ecalIsol+=ebc_bcet;
	    }
	  }
	}
//      } // matches ebc_bcchi2
    } // matches ebc_bcet && ebc_bcalgo

  }
  
  //  std::cout << "Will return ecalIsol = " << ecalIsol << std::endl; 
  return ecalIsol;
  
}

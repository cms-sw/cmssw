//*****************************************************************************
// File:      EgammaEcalExtractor.cc
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
#include "RecoEgamma/EgammaIsolationAlgos/plugins/EgammaEcalExtractor.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

using namespace egammaisolation;
using namespace reco::isodeposit;

EgammaEcalExtractor::~EgammaEcalExtractor(){}

reco::IsoDeposit EgammaEcalExtractor::deposit(const edm::Event & ev, const edm::EventSetup & evSetup, const reco::Candidate & candidate) const {
  edm::Handle<reco::SuperClusterCollection> superClusterCollectionH;
  edm::Handle<reco::BasicClusterCollection> basicClusterCollectionH;
  ev.getByToken(superClusterToken_, superClusterCollectionH);
  ev.getByToken(basicClusterToken_, basicClusterCollectionH);

  reco::SuperClusterRef sc = candidate.get<reco::SuperClusterRef>();
  math::XYZPoint position = sc->position();
  // match the photon hybrid supercluster with those with Algo==0 (island)
  double delta1=1000.;
  double deltacur=1000.;
  const reco::SuperCluster *matchedsupercluster=0;
  bool MATCHEDSC = false;

  Direction candDir(position.eta(), position.phi());
  reco::IsoDeposit deposit(candDir );
  deposit.setVeto( reco::IsoDeposit::Veto(candDir, 0) ); // no veto is needed for this deposit
  deposit.addCandEnergy(sc->energy()*sin(2*atan(exp(-sc->eta()))));

  for(reco::SuperClusterCollection::const_iterator scItr = superClusterCollectionH->begin(); scItr != superClusterCollectionH->end(); ++scItr){

    const reco::SuperCluster *supercluster = &(*scItr);

    if(supercluster->seed()->algo() == 0){
      deltacur = ROOT::Math::VectorUtil::DeltaR(supercluster->position(), position);
      if (deltacur < delta1) {
        delta1=deltacur;
	matchedsupercluster = supercluster;
	MATCHEDSC = true;
      }
    }
  }

  const reco::BasicCluster *cluster= 0;

  //loop over basic clusters
  for(reco::BasicClusterCollection::const_iterator cItr = basicClusterCollectionH->begin(); cItr != basicClusterCollectionH->end(); ++cItr){

    cluster = &(*cItr);
//    double ebc_bcchi2 = cluster->chi2();
    int    ebc_bcalgo = cluster->algo();
    double ebc_bce    = cluster->energy();
    double ebc_bceta  = cluster->eta();
    double ebc_bcet   = ebc_bce*sin(2*atan(exp(ebc_bceta)));
    double newDelta = 0.;

    if (ebc_bcet > etMin_ && ebc_bcalgo == 0) {
//      if (ebc_bcchi2 < 30.) {

	if(MATCHEDSC || !scmatch_ ){  //skip selection if user wants to fill all superclusters
	  bool inSuperCluster = false;

	  if( scmatch_ ){ // only try the matching if needed
	    reco::CaloCluster_iterator theEclust = matchedsupercluster->clustersBegin();
	    // loop over the basic clusters of the matched supercluster
	    for(;theEclust != matchedsupercluster->clustersEnd(); ++theEclust) {
	    if ((**theEclust) ==  (*cluster) ) inSuperCluster = true;
	    }
	  }
	  if (!inSuperCluster || !scmatch_ ) {  //skip selection if user wants to fill all superclusters
	    newDelta=ROOT::Math::VectorUtil::DeltaR(cluster->position(),position);
	    if(newDelta < conesize_) {
              deposit.addDeposit( Direction(cluster->eta(), cluster->phi()), ebc_bcet);
	    }
	  }
	}
//      } // matches ebc_bcchi2
    } // matches ebc_bcet && ebc_bcalgo

  }

  //  std::cout << "Will return ecalIsol = " << ecalIsol << std::endl;
  return deposit;

}

#ifndef ggPFESClusters_h
#define ggPFESClusters_h
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/PreshowerCluster.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include <memory>
using namespace edm;
using namespace std;
using namespace reco;

class ggPFESClusters{
  
 public:
  
  explicit ggPFESClusters(			   
			  edm::Handle<EcalRecHitCollection>& ESRecHits,
			  const CaloSubdetectorGeometry* geomEnd
			  );
  virtual ~ggPFESClusters();
  virtual vector<reco::PreshowerCluster>getPFESClusters(reco::SuperCluster sc);
  double getLinkDist(reco::PreshowerCluster clusterPS, reco::CaloCluster
		     clusterECAL);
 private:
  Handle<EcalRecHitCollection>ESRecHits_;
  const CaloSubdetectorGeometry* geomEnd_;
};
#endif

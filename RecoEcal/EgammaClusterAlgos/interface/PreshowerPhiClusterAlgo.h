#ifndef RecoEcal_EgammaClusterAlgos_PreshowerPhiClusterAlgo_h
#define RecoEcal_EgammaClusterAlgos_PreshowerPhiClusterAlgo_h

#include "DataFormats/EgammaReco/interface/PreshowerCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "RecoCaloTools/Navigation/interface/EcalPreshowerNavigator.h"
#include <string>
#include <vector>
#include <set>

class CaloSubdetectorGeometry;
class CaloSubdetectorTopology;

class PreshowerPhiClusterAlgo {
  
 public:
  
  typedef math::XYZPoint Point;
  
  typedef std::map<DetId, EcalRecHit> RecHitsMap;
  typedef std::set<DetId> HitsID;

  PreshowerPhiClusterAlgo() : 
    esStripEnergyCut_(0.)
    {}
    
  PreshowerPhiClusterAlgo(float stripEnergyCut) :
    esStripEnergyCut_(stripEnergyCut)
    {}
    
    ~PreshowerPhiClusterAlgo() {};
    
    reco::PreshowerCluster makeOneCluster(ESDetId strip,
					  HitsID *used_strips,
					  RecHitsMap *rechits_map,
					  const CaloSubdetectorGeometry*& geometry_p,
					  CaloSubdetectorTopology*& topology_p,
					  double deltaEta, double minDeltaPhi, double maxDeltaPhi);
    
    bool goodStrip(RecHitsMap::iterator candidate_it);
    
 private:
    
      float esStripEnergyCut_;
      
      std::vector<ESDetId> road_2d;
      
      // The map of hits
      RecHitsMap *rechits_map;
      
      // The set of used DetID's
      HitsID *used_s;
      
};
#endif


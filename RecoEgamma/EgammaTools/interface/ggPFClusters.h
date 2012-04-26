#ifndef ggPFClusters_h
#define ggPFClusters_h
#include "CondFormats/EgammaObjects/interface/GBRForest.h"
#include "DataFormats/PatCandidates/interface/Photon.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include <memory>
using namespace edm;
using namespace std;
using namespace reco;

class ggPFClusters  {
  
 public:
  
  explicit ggPFClusters(
			//		reco::Photon PFPhoton,
			edm::Handle<EcalRecHitCollection>& EBReducedRecHits,
			edm::Handle<EcalRecHitCollection>& EEReducedRecHits,
			const CaloSubdetectorGeometry* geomBar,
			const CaloSubdetectorGeometry* geomEnd
			);
  virtual ~ggPFClusters();
  //return PFClusters
  virtual vector<reco::CaloCluster>getPFClusters(reco::SuperCluster);
  //compute Energy
  virtual float SumPFRecHits(std::vector< std::pair<DetId, float> >& bcCells, bool isEB);
  // return the PFCluster Energy from Rec Hits that match SC Footprint
  virtual float getPFSuperclusterOverlap( reco::CaloCluster PFClust, reco::Photon phot);
  // compute the PFCluster Energy from Rec Hits that match SC Footprint
  virtual float PFRecHitsSCOverlap(
				   std::vector< std::pair<DetId, float> >& bcCells1, 
				   std::vector< std::pair<DetId, float> >& bcCells2,
				   bool isEB);
 
  //Local Coordinates for a PFCluster:
  virtual void localCoordsEB( reco::CaloCluster clus, float &etacry, float &phicry, int &ieta, int &iphi, float &thetatilt, float &phitilt);
  virtual void localCoordsEE(reco::CaloCluster clus, float &xcry, float &ycry, int &ix, int &iy, float &thetatilt, float &phitilt);
  
  //Cluster Shapes.
  virtual float get5x5Element(int i, int j,std::vector< std::pair<DetId, float> >& bcCells, bool isEB );
  virtual void Fill5x5Map(std::vector< std::pair<DetId, float> >& bcCells, bool isEB);
  virtual DetId FindSeed(std::vector< std::pair<DetId, float> >& bcCells, bool isEB);  
  virtual std::pair<float, float>ClusterWidth(vector<reco::CaloCluster>& PFClust);
  double LocalEnergyCorrection(const GBRForest *ReaderLCEB, const GBRForest *ReaderLCEE, reco::CaloCluster PFClust, float beamspotZ);
 private:
 Handle<EcalRecHitCollection>  EBReducedRecHits_;
 Handle<EcalRecHitCollection>  EEReducedRecHits_;
 const CaloSubdetectorGeometry* geomBar_;
 const CaloSubdetectorGeometry* geomEnd_;
 float e5x5_[5][5];
};
#endif

#ifndef RecoLocalCalo_HGCalRecAlgos_HGCalClusteringAlgoBase_h
#define RecoLocalCalo_HGCalRecAlgos_HGCalClusteringAlgoBase_h

#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

// C/C++ headers
#include <vector>

class HGCalClusteringAlgoBase
{

public:

enum VerbosityLevel { pDEBUG = 0, pWARNING = 1, pINFO = 2, pERROR = 3 };

 HGCalClusteringAlgoBase(VerbosityLevel v,
     reco::CaloCluster::AlgoId algo)
   : verbosity_(v), algoId_(algo) {};
 virtual ~HGCalClusteringAlgoBase() {}

 virtual void populate(const HGCRecHitCollection &hits) = 0;
 virtual void makeClusters() = 0;
 virtual std::vector<reco::BasicCluster> getClusters(bool) = 0;
 virtual void reset() = 0;

 inline void getEventSetup(const edm::EventSetup& es){
   rhtools_.getEventSetup(es);
 }
 inline void setVerbosity(VerbosityLevel the_verbosity) {
   verbosity_ = the_verbosity;
 }

 //max number of layers
 static const unsigned int maxlayer = 52;
 // last layer per subdetector
 static const unsigned int lastLayerEE = 28;
 static const unsigned int lastLayerFH = 40;

protected:
 // The verbosity level
 VerbosityLevel verbosity_;

 // The vector of clusters
 std::vector<reco::BasicCluster> clusters_v_;

 hgcal::RecHitTools rhtools_;

 // The algo id
 reco::CaloCluster::AlgoId algoId_;

};

#endif

#ifndef RecoLocalCalo_HGCalRecProducers_HGCalClusteringAlgoBase_h
#define RecoLocalCalo_HGCalRecProducers_HGCalClusteringAlgoBase_h

#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

// C/C++ headers
#include <vector>
#include <numeric>

namespace hgcal_clustering {
template <typename T>
std::vector<size_t> sorted_indices(const std::vector<T> &v) {

        // initialize original index locations
        std::vector<size_t> idx(v.size());
        std::iota (std::begin(idx), std::end(idx), 0);

        // sort indices based on comparing values in v
        std::sort(idx.begin(), idx.end(),
                  [&v](size_t i1, size_t i2) {
                return v[i1] > v[i2];
        });

        return idx;
}

template <typename T>
size_t max_index(const std::vector<T> &v) {

        // initialize original index locations
        std::vector<size_t> idx(v.size(),0);
        std::iota (std::begin(idx), std::end(idx), 0);

        // take the max index based on comparing values in v
        auto maxidx = std::max_element(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1].data.rho < v[i2].data.rho;});

        return (*maxidx);
}

//Density collection
typedef std::map< DetId, float > Density;

};

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
 virtual hgcal_clustering::Density getDensity() = 0;

 inline void getEventSetup(const edm::EventSetup& es){
   rhtools_.getEventSetup(es);
 }
 inline void setVerbosity(VerbosityLevel the_verbosity) {
   verbosity_ = the_verbosity;
 }
 inline void setAlgoId(reco::CaloCluster::AlgoId algo) {algoId_ = algo;}

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

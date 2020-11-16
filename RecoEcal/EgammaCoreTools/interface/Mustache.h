#ifndef RecoEcal_EgammaCoreTools_Mustache_h
#define RecoEcal_EgammaCoreTools_Mustache_h

#include <vector>
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "CondFormats/EcalObjects/interface/EcalMustacheSCParameters.h"
#include "CondFormats/EcalObjects/interface/EcalSCDynamicDPhiParameters.h"

namespace reco {
  namespace MustacheKernel {
    bool inMustache(const EcalMustacheSCParameters* params,
                    const float maxEta,
                    const float maxPhi,
                    const float ClustE,
                    const float ClusEta,
                    const float ClusPhi);
    bool inDynamicDPhiWindow(const EcalSCDynamicDPhiParameters* params,
                             const float seedEta,
                             const float seedPhi,
                             const float ClustE,
                             const float ClusEta,
                             const float clusPhi);

  }  // namespace MustacheKernel

  class Mustache {
  public:
    Mustache(const EcalMustacheSCParameters* mustache_params);

    void MustacheID(const CaloClusterPtrVector& clusters, int& nclusters, float& EoutsideMustache);
    void MustacheID(const std::vector<const CaloCluster*>&, int& nclusers, float& EoutsideMustache);
    void MustacheID(const reco::SuperCluster& sc, int& nclusters, float& EoutsideMustache);

    void MustacheClust(const std::vector<CaloCluster>& clusters,
                       std::vector<unsigned int>& insideMust,
                       std::vector<unsigned int>& outsideMust);

    void FillMustacheVar(const std::vector<CaloCluster>& clusters);
    //return Functions for Mustache Variables:
    float MustacheE() { return Energy_In_Mustache_; }
    float MustacheEOut() { return Energy_Outside_Mustache_; }
    float MustacheEtOut() { return Et_Outside_Mustache_; }
    float LowestMustClust() { return LowestClusterEInMustache_; }
    int InsideMust() { return included_; }
    int OutsideMust() { return excluded_; }

  private:
    template <class RandomAccessPtrIterator>
    void MustacheID(const RandomAccessPtrIterator&,
                    const RandomAccessPtrIterator&,
                    int& nclusters,
                    float& EoutsideMustache);

    float Energy_In_Mustache_;
    float Energy_Outside_Mustache_;
    float Et_Outside_Mustache_;
    float LowestClusterEInMustache_;
    int excluded_;
    int included_;
    const EcalMustacheSCParameters* mustache_params_;
  };

}  // namespace reco

#endif

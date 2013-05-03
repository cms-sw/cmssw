#ifndef RecoEcal_EgammaCoreTools_Mustache_h
#define RecoEcal_EgammaCoreTools_Mustache_h

#include <vector>
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"



namespace reco {
  namespace MustacheKernel {    
      bool inMustache(float maxEta, float maxPhi, 
		      float ClustE, float ClusEta, float ClusPhi);
     
  }

  class Mustache {
    
  public:
    void MustacheID(CaloClusterPtrVector& clusters, int & nclusters, float & EoutsideMustache);
    void MustacheID(std::vector<const CaloCluster*>&, int & nclusers,float & EoutsideMustache); 
    void MustacheID(const reco::SuperCluster& sc, int & nclusters, float & EoutsideMustache);
    void MustacheClust(std::vector<CaloCluster>& clusters, std::vector<unsigned int>& insideMust, std::vector<unsigned int>& outsideMust);
    
    void FillMustacheVar(std::vector<CaloCluster>& clusters);
    //return Functions for Mustache Variables:
    float MustacheE(){return Energy_In_Mustache_;}
    float MustacheEOut(){return Energy_Outside_Mustache_;}
    float MustacheEtOut(){return Et_Outside_Mustache_;}
    float LowestMustClust(){return LowestClusterEInMustache_;}
    int InsideMust(){return included_;}
    int OutsideMust(){return excluded_;}
  private:
    float Energy_In_Mustache_;
    float Energy_Outside_Mustache_;
    float Et_Outside_Mustache_;
    float LowestClusterEInMustache_;
    int excluded_;
    int included_;
  };

  
}

#endif

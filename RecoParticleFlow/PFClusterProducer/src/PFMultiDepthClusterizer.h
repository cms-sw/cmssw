#ifndef __PFMultiDepthClusterizer_H__
#define __PFMultiDepthClusterizer_H__

#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterBuilderBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"

#include <unordered_map>

class PFMultiDepthClusterizer : public PFClusterBuilderBase {
  typedef PFMultiDepthClusterizer B2DGPF;
 public:
  PFMultiDepthClusterizer(const edm::ParameterSet& conf);
    
  virtual ~PFMultiDepthClusterizer() {}
  PFMultiDepthClusterizer(const B2DGPF&) = delete;
  B2DGPF& operator=(const B2DGPF&) = delete;

  void update(const edm::EventSetup& es) { 
    _allCellsPosCalc->update(es);
  }

  void buildClusters(const reco::PFClusterCollection&,
		     const std::vector<bool>&,
		     reco::PFClusterCollection& outclus);

 private:  
  std::unique_ptr<PFCPositionCalculatorBase> _allCellsPosCalc;
  double  nSigmaEta_;
  double  nSigmaPhi_;
  

  class ClusterLink {
  public:
    ClusterLink(unsigned int i,unsigned int j,double DR,double DZ,double energy) {
      from_ = i;
      to_ = j;
      linkDR_ =DR; 
      linkDZ_ =DZ; 
      linkE_ =energy; 
    }

    ~ClusterLink() {

    }

    unsigned int from() const {return from_;}
    unsigned int to() const {return to_;}
    double dR() const {return linkDR_;}
    double dZ() const {return linkDZ_;}
    double energy() const {return linkE_;}
    
  private:
    unsigned int from_;
    unsigned int to_;
    double linkDR_;
    double linkDZ_;
    double linkE_;

  };


  void calculateShowerShapes(const reco::PFClusterCollection&,std::vector<double>&,std::vector<double>&); 
  std::vector<ClusterLink> link(const reco::PFClusterCollection&,const std::vector<double>&,const std::vector<double>&);
  std::vector<ClusterLink>  prune(std::vector<ClusterLink>&,std::vector<bool>& linkedClusters);


  void expandCluster(reco::PFCluster&,unsigned int point, std::vector<bool>& mask,const reco::PFClusterCollection& , const std::vector<ClusterLink>& links);

  void absorbCluster(reco::PFCluster&, const reco::PFCluster&) ;



};

DEFINE_EDM_PLUGIN(PFClusterBuilderFactory,
		  PFMultiDepthClusterizer,
		  "PFMultiDepthClusterizer");

#endif

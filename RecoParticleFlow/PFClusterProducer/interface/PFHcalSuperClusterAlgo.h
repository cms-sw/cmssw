#ifndef RecoParticleFlow_PFClusterProducer_PFHcalSuperClusterAlgo_h
#define RecoParticleFlow_PFClusterProducer_PFHcalSuperClusterAlgo_h

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFSuperCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFSuperClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

#include <vector>

#include <memory>

/// \brief Algorithm for particle flow superclustering
/*!
  This class takes as an input a map of pointers to PFCluster's, and creates 
  PFSuperCluster's from these clusters.

  \todo describe algorithm and parameters. give a use case

  \author Chris Tully
  \date July 2012
*/
class PFHcalSuperClusterAlgo {

 public:

  /// constructor
  PFHcalSuperClusterAlgo();

  /// destructor
  virtual ~PFHcalSuperClusterAlgo() {;}

  /// enable/disable debugging
  void enableDebugging(bool debug) { debug_ = debug;}
 

  typedef edm::Handle< reco::PFClusterCollection > PFClusterHandle;
  typedef edm::Ptr< reco::PFCluster> PFClusterPtr;
  const edm::PtrVector<reco::PFCluster>& clusters() const
  {return clusters_; }
  
  /// perform clustering
  void doClustering( const reco::PFClusterCollection& clusters, const reco::PFClusterCollection& clustersHO );

 /// calculate eta-phi widths of clusters
  std::pair<double, double> calculateWidths(const reco::PFCluster& cluster);

 /// recalculate eta-phi position of clusters
  std::pair<double, double> calculatePosition(const reco::PFCluster& cluster);

  /// perform clustering in full framework
  void doClustering( const PFClusterHandle& clustersHandle, const PFClusterHandle& clustersHOHandle );

  // set histogram file pointer
//  void setHistos(TFile* file);

  // write histos
  void write();
  
  /// setters -------------------------------------------------------
  
  /// getters -------------------------------------------------------
 
  /// \return particle flow clusters
  std::auto_ptr< std::vector< reco::PFCluster > >& clusters()  
    {return pfClusters_;}

  /// \return particle flow superclusters
  std::auto_ptr< std::vector< reco::PFSuperCluster > >& superClusters()  
    {return pfSuperClusters_;}


  friend std::ostream& operator<<(std::ostream& out,const PFHcalSuperClusterAlgo& algo);


 private:
  /// perform clustering
  void doClusteringWorker( const reco::PFClusterCollection& clusters, const reco::PFClusterCollection& clustersHO );

  /// particle flow clusters
  std::auto_ptr< std::vector<reco::PFCluster> > pfClusters_;
  
  /// particle flow superclusters
  std::auto_ptr< std::vector<reco::PFSuperCluster> > pfSuperClusters_;

  reco::PFSuperCluster            SuperCluster_; 
  PFClusterHandle           clustersHandle_;
  PFClusterHandle           clustersHOHandle_;
  edm::PtrVector< reco::PFCluster >  clusters_;


  /// debugging on/off
  bool   debug_;

  /// product number 
  static  unsigned  prodNum_;

};

#endif

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

  

  \author Chris Tully (updated by Josh Kaisen)
  \date July 2012 (April 2013)
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

 /// recalculate eta-phi position of clusters or their widths
  std::pair<double, double> calcPosition(const reco::PFCluster& cluster);
 //Calculate widths (delta phi or eta)^2 in order to check range for clustering
  std::pair<double, double> calcWidth(const reco::PFCluster& cluster, std::pair<double, double> Position);

  //Describes whether a cluster is in range of another cluster to be merged, can be changed so that it gives a value to decide which cluster merges more accurately with another cluster. 
  std::pair<bool, double> TestMerger( std::pair<double,double> Position1, std::pair<double,double> Width1, std::pair<double,double> Position2, std::pair<double,double> Width2);

  //Forms links between clusters close enough to be merged into superclusters
  void FormLinks( std::vector< bool >& lmerge, std::vector<unsigned>& iroot, std::vector<double>& idR, unsigned d, unsigned ic, std::map<unsigned, std::vector<std::pair< reco::PFCluster,unsigned > > >& clustersbydepth, std::pair<double,double> Position, std::pair<double,double> Width );

  //Merges clusters that have links between them
  void MergeClusters( std::vector< bool >& lmerge, std::vector<unsigned>& iroot, std::vector< bool >& lmergeHO, std::vector<unsigned>& irootHO, unsigned depthHO, unsigned d, unsigned ic, std::map<unsigned, std::vector<std::pair< reco::PFCluster,unsigned > > >& clustersbydepth, std::map<unsigned, std::vector<std::pair< reco::PFCluster,unsigned > > >& clustersbydepthHO, edm::PtrVector< reco::PFCluster >& mergeclusters, bool HOSupercluster );

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

  /// \return particle flow clusters
  std::auto_ptr< std::vector< reco::PFCluster > >& clustersHO()  
    {return pfClustersHO_;}

  /// \return particle flow superclusters
  std::auto_ptr< std::vector< reco::PFSuperCluster > >& superClustersHO()  
    {return pfSuperClustersHO_;}


  friend std::ostream& operator<<(std::ostream& out,const PFHcalSuperClusterAlgo& algo);


 private:
  /// perform clustering
  void doClusteringWorker( const reco::PFClusterCollection& clusters, const reco::PFClusterCollection& clustersHO );

  /// particle flow clusters
  std::auto_ptr< std::vector<reco::PFCluster> > pfClusters_;
  
  /// particle flow superclusters
  std::auto_ptr< std::vector<reco::PFSuperCluster> > pfSuperClusters_;

  // particle flow clusters and superclusters with HO in them
  std::auto_ptr< std::vector<reco::PFCluster> > pfClustersHO_;
    
  std::auto_ptr< std::vector<reco::PFSuperCluster> > pfSuperClustersHO_;

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

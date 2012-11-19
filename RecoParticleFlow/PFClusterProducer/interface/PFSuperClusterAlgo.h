#ifndef RecoParticleFlow_PFClusterProducer_PFSuperClusterAlgo_h
#define RecoParticleFlow_PFClusterProducer_PFSuperClusterAlgo_h

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFSuperCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFSuperClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

#include <string>
#include <vector>
#include <map>
#include <set>

#include <memory>

class TFile;
class TH1F;
class TH2F;

/// \brief Algorithm for particle flow superclustering
/*!
  This class takes as an input a map of pointers to PFCluster's, and creates 
  PFSuperCluster's from these clusters.

  \todo describe algorithm and parameters. give a use case

  \author Chris Tully
  \date July 2012
*/
class PFSuperClusterAlgo {

 public:

  /// constructor
  PFSuperClusterAlgo();

  /// destructor
  virtual ~PFSuperClusterAlgo() {;}

  /// enable/disable debugging
  void enableDebugging(bool debug) { debug_ = debug;}
 

  typedef edm::Handle< reco::PFClusterCollection > PFClusterHandle;
  
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


  friend std::ostream& operator<<(std::ostream& out,const PFSuperClusterAlgo& algo);


 private:
  /// perform clustering
  void doClusteringWorker( const reco::PFClusterCollection& clusters, const reco::PFClusterCollection& clustersHO );

  /// particle flow clusters
  std::auto_ptr< std::vector<reco::PFCluster> > pfClusters_;
  
  /// particle flow superclusters
  std::auto_ptr< std::vector<reco::PFSuperCluster> > pfSuperClusters_;
  
  PFClusterHandle           clustersHandle_;
  PFClusterHandle           clustersHOHandle_;

  /// debugging on/off
  bool   debug_;

  /// product number 
  static  unsigned  prodNum_;

  // Histograms
  TH1F* dR12HB;
  TH1F* dR13HB;
  TH1F* dR23HB;
  TH1F* dR3HOHB;
  TH1F* dR12HE;
  TH1F* dR13HE;
  TH1F* dR23HE;
  TH1F* dR24HE;
  TH1F* dR34HE;
  TH1F* dR35HE;
  TH1F* dR45HE;
  TH1F* dEta12HB;
  TH1F* dEta13HB;
  TH1F* dEta23HB;
  TH1F* dEta3HOHB;
  TH1F* dEta12HE;
  TH1F* dEta13HE;
  TH1F* dEta23HE;
  TH1F* dEta24HE;
  TH1F* dEta34HE;
  TH1F* dEta35HE;
  TH1F* dEta45HE;
  TH1F* dPhi12HB;
  TH1F* dPhi13HB;
  TH1F* dPhi23HB;
  TH1F* dPhi3HOHB;
  TH1F* dPhi12HE;
  TH1F* dPhi13HE;
  TH1F* dPhi23HE;
  TH1F* dPhi24HE;
  TH1F* dPhi34HE;
  TH1F* dPhi35HE;
  TH1F* dPhi45HE;
  TH1F* normalized12HB;
  TH1F* normalized13HB;
  TH1F* normalized23HB;
  TH1F* normalized3HOHB;
  TH1F* normalized12HE;
  TH1F* normalized13HE;
  TH1F* normalized23HE;
  TH1F* normalized24HE;
  TH1F* normalized34HE;
  TH1F* normalized35HE;
  TH1F* normalized45HE;
  TH1F* nclustersHB;
  TH1F* nclustersHE;
  TH1F* nclustersHO;
  TH1F* mergeclusters1HB;
  TH1F* mergeclusters2HB;
  TH1F* mergeclusters3HB;
  TH1F* mergeclusters1HE;
  TH1F* mergeclusters2HE;
  TH1F* mergeclusters3HE;
  TH1F* mergeclusters4HE;
  TH1F* hitsHB;
  TH1F* hitsHE;
  TH2F* etaPhi;
  TH2F* etaPhiHits;
  TH2F* etaPhiHits1HB;
  TH2F* etaPhiHits2HB;
  TH2F* etaPhiHits3HB;
  TH2F* etaPhiHits1HE;
  TH2F* etaPhiHits2HE;
  TH2F* etaPhiHits3HE;
  TH2F* etaPhiHits4HE;
  TH2F* etaPhiHits5HE;
  TH1F* hitTime1HB;
  TH1F* hitTime2HB;
  TH1F* hitTime3HB;
  TH1F* hitTime1HE;
  TH1F* hitTime2HE;
  TH1F* hitTime3HE;
  TH1F* hitTime4HE;
  TH1F* hitTime5HE;
  TH1F* etaWidth1HB;
  TH1F* etaWidth2HB;
  TH1F* etaWidth3HB;
  TH1F* etaWidth1HE;
  TH1F* etaWidth2HE;
  TH1F* etaWidth3HE;
  TH1F* etaWidth4HE;
  TH1F* etaWidth5HE;
  TH1F* phiWidth1HB;
  TH1F* phiWidth2HB;
  TH1F* phiWidth3HB;
  TH1F* phiWidth1HE;
  TH1F* phiWidth2HE;
  TH1F* phiWidth3HE;
  TH1F* phiWidth4HE;
  TH1F* phiWidth5HE;
  TH1F* etaWidthSuperClusterHB;
  TH1F* phiWidthSuperClusterHB;
  TH1F* etaWidthSuperClusterHE;
  TH1F* phiWidthSuperClusterHE;
  TH1F* sizeSuperClusterHB;
  TH1F* sizeSuperClusterHE;  
TFile*     file_;

};

#endif

#include "RecoParticleFlow/PFClusterProducer/plugins/PFClusterProducer.h"

#include <memory>

#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterAlgo.h"

#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"



using namespace std;
using namespace edm;

PFClusterProducer::PFClusterProducer(const edm::ParameterSet& iConfig)
{
    
  verbose_ = 
    iConfig.getUntrackedParameter<bool>("verbose",false);



  // parameters for clustering
  
  double threshBarrel = 
    iConfig.getParameter<double>("thresh_Barrel");
  double threshSeedBarrel = 
    iConfig.getParameter<double>("thresh_Seed_Barrel");

  double threshPtBarrel = 
    iConfig.getParameter<double>("thresh_Pt_Barrel");
  double threshPtSeedBarrel = 
    iConfig.getParameter<double>("thresh_Pt_Seed_Barrel");

  double threshEndcap = 
    iConfig.getParameter<double>("thresh_Endcap");
  double threshSeedEndcap = 
    iConfig.getParameter<double>("thresh_Seed_Endcap");

  double threshPtEndcap = 
    iConfig.getParameter<double>("thresh_Pt_Endcap");
  double threshPtSeedEndcap = 
    iConfig.getParameter<double>("thresh_Pt_Seed_Endcap");


  int nNeighbours = 
    iConfig.getParameter<int>("nNeighbours");

//   double posCalcP1 = 
//     iConfig.getParameter<double>("posCalcP1");

  int posCalcNCrystal = 
    iConfig.getParameter<int>("posCalcNCrystal");
    
  double showerSigma = 
    iConfig.getParameter<double>("showerSigma");
    
  bool useCornerCells =
    iConfig.getParameter<bool>("useCornerCells");


  clusterAlgo_.setThreshBarrel( threshBarrel );
  clusterAlgo_.setThreshSeedBarrel( threshSeedBarrel );
  
  clusterAlgo_.setThreshPtBarrel( threshPtBarrel );
  clusterAlgo_.setThreshPtSeedBarrel( threshPtSeedBarrel );
  
  clusterAlgo_.setThreshEndcap( threshEndcap );
  clusterAlgo_.setThreshSeedEndcap( threshSeedEndcap );

  clusterAlgo_.setThreshPtEndcap( threshPtEndcap );
  clusterAlgo_.setThreshPtSeedEndcap( threshPtSeedEndcap );

  clusterAlgo_.setNNeighbours( nNeighbours );

  // p1 set to the minimum rechit threshold:
  double posCalcP1 = threshBarrel<threshEndcap ? threshBarrel:threshEndcap;
  clusterAlgo_.setPosCalcP1( posCalcP1 );
  clusterAlgo_.setPosCalcNCrystal( posCalcNCrystal );
  clusterAlgo_.setShowerSigma( showerSigma );

  clusterAlgo_.setUseCornerCells( useCornerCells  );


  int dcormode = 
    iConfig.getParameter<int>("depthCor_Mode");
  
  double dcora = 
    iConfig.getParameter<double>("depthCor_A");
  double dcorb = 
    iConfig.getParameter<double>("depthCor_B");
  double dcorap = 
    iConfig.getParameter<double>("depthCor_A_preshower");
  double dcorbp = 
    iConfig.getParameter<double>("depthCor_B_preshower");

  if( dcormode !=0 )
    reco::PFCluster::setDepthCorParameters( dcormode, 
					    dcora, dcorb, 
					    dcorap, dcorbp );






  // access to the collections of rechits from the various detectors:

  
  inputTagPFRecHits_ = 
    iConfig.getParameter<InputTag>("PFRecHits");
  //---ab

  //inputTagClusterCollectionName_ =  iConfig.getParameter<string>("PFClusterCollectionName");    
 
  // produces<reco::PFClusterCollection>(inputTagClusterCollectionName_);
   produces<reco::PFClusterCollection>();
    //---ab
}



PFClusterProducer::~PFClusterProducer() {}




void PFClusterProducer::produce(edm::Event& iEvent, 
				const edm::EventSetup& iSetup) {
  

  edm::Handle< reco::PFRecHitCollection > rechitsHandle;
  
  // access the rechits in the event
  bool found = iEvent.getByLabel( inputTagPFRecHits_, rechitsHandle );  

  if(!found ) {

    ostringstream err;
    err<<"cannot find rechits: "<<inputTagPFRecHits_;
    LogError("PFClusterProducer")<<err.str()<<endl;
    
    throw cms::Exception( "MissingProduct", err.str());
  }


  // do clustering
  clusterAlgo_.doClustering( rechitsHandle );
  
  if( verbose_ ) {
    LogInfo("PFClusterProducer")
      <<"  clusters --------------------------------- "<<endl
      <<clusterAlgo_<<endl;
  }    
  
  // get clusters out of the clustering algorithm 
  // and put them in the event. There is no copy.
  auto_ptr< vector<reco::PFCluster> > 
    outClusters( clusterAlgo_.clusters() ); 
  iEvent.put( outClusters );    

}
  



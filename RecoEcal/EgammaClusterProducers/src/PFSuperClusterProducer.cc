#include "RecoEcal/EgammaClusterProducers/interface/PFSuperClusterProducer.h"

#include <memory>

#include "RecoEcal/EgammaClusterAlgos/interface/PFSuperClusterAlgo.h"

#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

using namespace std;
using namespace edm;

PFSuperClusterProducer::PFSuperClusterProducer(const edm::ParameterSet& iConfig)
{
    
  verbose_ = 
    iConfig.getUntrackedParameter<bool>("verbose",false);

  

  // parameters for clustering
  
  double threshPFClusterSeedBarrel = iConfig.getParameter<double>("thresh_PFClusterSeedBarrel");
  double threshPFClusterBarrel = iConfig.getParameter<double>("thresh_PFClusterBarrel");

  double threshPFClusterSeedEndcap = iConfig.getParameter<double>("thresh_PFClusterSeedEndcap");
  double threshPFClusterEndcap = iConfig.getParameter<double>("thresh_PFClusterEndcap");
  
  double phiwidthSuperClusterBarrel = iConfig.getParameter<double>("phiwidth_SuperClusterBarrel");
  double etawidthSuperClusterBarrel = iConfig.getParameter<double>("etawidth_SuperClusterBarrel");

  double phiwidthSuperClusterEndcap = iConfig.getParameter<double>("phiwidth_SuperClusterEndcap");
  double etawidthSuperClusterEndcap = iConfig.getParameter<double>("etawidth_SuperClusterEndcap");

  double threshPFClusterES = iConfig.getParameter<double>("thresh_PFClusterES");

  //double threshPFClusterMustacheOutBarrel = iConfig.getParameter<double>("thresh_PFClusterMustacheOutBarrel");
  //double threshPFClusterMustacheOutEndcap = iConfig.getParameter<double>("thresh_PFClusterMustacheOutEndcap");

  double doMustacheCut = iConfig.getParameter<bool>("doMustachePUcleaning");

  superClusterAlgo_.setVerbosityLevel(verbose_);

  superClusterAlgo_.setThreshPFClusterSeedBarrel( threshPFClusterSeedBarrel );
  superClusterAlgo_.setThreshPFClusterBarrel( threshPFClusterBarrel );

  superClusterAlgo_.setThreshPFClusterSeedEndcap( threshPFClusterSeedEndcap );
  superClusterAlgo_.setThreshPFClusterEndcap( threshPFClusterEndcap );

  superClusterAlgo_.setPhiwidthSuperClusterBarrel( phiwidthSuperClusterBarrel );
  superClusterAlgo_.setEtawidthSuperClusterBarrel( etawidthSuperClusterBarrel );

  superClusterAlgo_.setPhiwidthSuperClusterEndcap( phiwidthSuperClusterEndcap );
  superClusterAlgo_.setEtawidthSuperClusterEndcap( etawidthSuperClusterEndcap );

  superClusterAlgo_.setThreshPFClusterES( threshPFClusterES );

  superClusterAlgo_.setMustacheCut( doMustacheCut );
  //superClusterAlgo_.setThreshPFClusterMustacheOutBarrel( threshPFClusterMustacheOutBarrel );
  //superClusterAlgo_.setThreshPFClusterMustacheOutEndcap( threshPFClusterMustacheOutEndcap );


  bool applyCrackCorrections_ = iConfig.getParameter<bool>("applyCrackCorrections");
  superClusterAlgo_.setCrackCorrections(applyCrackCorrections_);


  
  inputTagPFClusters_ = iConfig.getParameter<InputTag>("PFClusters");
  inputTagPFClustersES_ = iConfig.getParameter<InputTag>("PFClustersES");

  PFBasicClusterCollectionBarrel_ = iConfig.getParameter<string>("PFBasicClusterCollectionBarrel");
  PFSuperClusterCollectionBarrel_ = iConfig.getParameter<string>("PFSuperClusterCollectionBarrel");

  PFBasicClusterCollectionEndcap_ = iConfig.getParameter<string>("PFBasicClusterCollectionEndcap");
  PFSuperClusterCollectionEndcap_ = iConfig.getParameter<string>("PFSuperClusterCollectionEndcap");

  PFBasicClusterCollectionPreshower_ = iConfig.getParameter<string>("PFBasicClusterCollectionPreshower");
  PFSuperClusterCollectionEndcapWithPreshower_ = iConfig.getParameter<string>("PFSuperClusterCollectionEndcapWithPreshower");

  produces<reco::BasicClusterCollection>(PFBasicClusterCollectionBarrel_);
  produces<reco::SuperClusterCollection>(PFSuperClusterCollectionBarrel_);
  produces<reco::BasicClusterCollection>(PFBasicClusterCollectionEndcap_);
  produces<reco::SuperClusterCollection>(PFSuperClusterCollectionEndcap_);
  //produces<reco::BasicClusterCollection>(PFBasicClusterCollectionPreshower_);
  produces<reco::SuperClusterCollection>(PFSuperClusterCollectionEndcapWithPreshower_);

   
}



PFSuperClusterProducer::~PFSuperClusterProducer() {}




void PFSuperClusterProducer::produce(edm::Event& iEvent, 
				const edm::EventSetup& iSetup) {
  

  //Load the pfcluster collections
  edm::Handle<reco::PFClusterCollection> pfclustersHandle;
  iEvent.getByLabel( inputTagPFClusters_, pfclustersHandle );  

  edm::Handle<reco::PFClusterCollection> preshowerpfclustersHandle;
  iEvent.getByLabel( inputTagPFClustersES_,  preshowerpfclustersHandle);


  //Load the ECAL energy calibration
  boost::shared_ptr<PFEnergyCalibration> thePFEnergyCalibration_(new PFEnergyCalibration());


  // do BARREL clustering 

  std::auto_ptr< reco::BasicClusterCollection >outBasicClustersBarrel (new reco::BasicClusterCollection);
  superClusterAlgo_.doClustering( pfclustersHandle, outBasicClustersBarrel, thePFEnergyCalibration_ , 0 );
  //cout << "doBarrelClustering done"<<endl; 

  const edm::OrphanHandle<reco::BasicClusterCollection> bcRefProdBarrel = iEvent.put(outBasicClustersBarrel,PFBasicClusterCollectionBarrel_);
  //cout << "outBasicClusters are put in the event" << endl;

  auto_ptr< reco::SuperClusterCollection > outSuperClustersBarrel(new reco::SuperClusterCollection);
  superClusterAlgo_.storeSuperClusters( bcRefProdBarrel, outSuperClustersBarrel );
  
  const edm::OrphanHandle<reco::SuperClusterCollection> scRefProdBarrel = iEvent.put(outSuperClustersBarrel, PFSuperClusterCollectionBarrel_);
  //cout << "outSuperClusters are put in the event" << endl;


  //do ENDCAP clustering

  std::auto_ptr< reco::BasicClusterCollection >outBasicClustersEndcap (new reco::BasicClusterCollection);
  superClusterAlgo_.doClustering( pfclustersHandle, outBasicClustersEndcap, thePFEnergyCalibration_ , 1);
  //cout << "doBarrelClustering done"<<endl;

  const edm::OrphanHandle<reco::BasicClusterCollection> bcRefProdEndcap = iEvent.put(outBasicClustersEndcap,PFBasicClusterCollectionEndcap_);
  //cout << "outBasicClusters are put in the event" << endl;

  auto_ptr< reco::SuperClusterCollection > outSuperClustersEndcap(new reco::SuperClusterCollection);
  superClusterAlgo_.storeSuperClusters( bcRefProdEndcap, outSuperClustersEndcap );
  
  const edm::OrphanHandle<reco::SuperClusterCollection> scRefProdEndcap = iEvent.put(outSuperClustersEndcap, PFSuperClusterCollectionEndcap_);
  //cout << "outSuperClusters are put in the event" << endl;

  
  auto_ptr< reco::SuperClusterCollection > outSuperClustersEndcapWithPreshower(new reco::SuperClusterCollection);
  superClusterAlgo_.matchSCtoESclusters(preshowerpfclustersHandle, outSuperClustersEndcapWithPreshower, thePFEnergyCalibration_, 1);
 
  const edm::OrphanHandle<reco::SuperClusterCollection> scRefProdEndcapWithPreshower = iEvent.put(outSuperClustersEndcapWithPreshower, PFSuperClusterCollectionEndcapWithPreshower_);


}
  



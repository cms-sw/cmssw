#include "RecoParticleFlow/PFClusterProducer/plugins/PFClusterProducer.h"

#include <memory>

#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterAlgo.h"

#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

// Geometry
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"

using namespace std;
using namespace edm;

namespace {
  const std::string PositionCalcType__EGPositionCalc("EGPositionCalc");
  const std::string PositionCalcType__EGPositionFormula("EGPositionFormula");
  const std::string PositionCalcType__PFPositionCalc("PFPositionCalc");
}

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

  double threshCleanBarrel = 
    iConfig.getParameter<double>("thresh_Clean_Barrel");
  std::vector<double> minS4S1CleanBarrel = 
    iConfig.getParameter< std::vector<double> >("minS4S1_Clean_Barrel");

  double threshEndcap = 
    iConfig.getParameter<double>("thresh_Endcap");
  double threshSeedEndcap = 
    iConfig.getParameter<double>("thresh_Seed_Endcap");

  double threshPtEndcap = 
    iConfig.getParameter<double>("thresh_Pt_Endcap");
  double threshPtSeedEndcap = 
    iConfig.getParameter<double>("thresh_Pt_Seed_Endcap");

  double threshCleanEndcap = 
    iConfig.getParameter<double>("thresh_Clean_Endcap");
  std::vector<double> minS4S1CleanEndcap = 
    iConfig.getParameter< std::vector<double> >("minS4S1_Clean_Endcap");

  double threshDoubleSpikeBarrel = 
    iConfig.getParameter<double>("thresh_DoubleSpike_Barrel");
  double minS6S2DoubleSpikeBarrel = 
    iConfig.getParameter<double>("minS6S2_DoubleSpike_Barrel");
  double threshDoubleSpikeEndcap = 
    iConfig.getParameter<double>("thresh_DoubleSpike_Endcap");
  double minS6S2DoubleSpikeEndcap = 
    iConfig.getParameter<double>("minS6S2_DoubleSpike_Endcap");

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

  bool cleanRBXandHPDs =
    iConfig.getParameter<bool>("cleanRBXandHPDs");


  clusterAlgo_.setThreshBarrel( threshBarrel );
  clusterAlgo_.setThreshSeedBarrel( threshSeedBarrel );
  
  clusterAlgo_.setThreshPtBarrel( threshPtBarrel );
  clusterAlgo_.setThreshPtSeedBarrel( threshPtSeedBarrel );
  
  clusterAlgo_.setThreshCleanBarrel(threshCleanBarrel);
  clusterAlgo_.setS4S1CleanBarrel(minS4S1CleanBarrel);

  clusterAlgo_.setThreshDoubleSpikeBarrel( threshDoubleSpikeBarrel );
  clusterAlgo_.setS6S2DoubleSpikeBarrel( minS6S2DoubleSpikeBarrel );

  clusterAlgo_.setThreshEndcap( threshEndcap );
  clusterAlgo_.setThreshSeedEndcap( threshSeedEndcap );

  clusterAlgo_.setThreshPtEndcap( threshPtEndcap );
  clusterAlgo_.setThreshPtSeedEndcap( threshPtSeedEndcap );

  clusterAlgo_.setThreshCleanEndcap(threshCleanEndcap);
  clusterAlgo_.setS4S1CleanEndcap(minS4S1CleanEndcap);

  clusterAlgo_.setThreshDoubleSpikeEndcap( threshDoubleSpikeEndcap );
  clusterAlgo_.setS6S2DoubleSpikeEndcap( minS6S2DoubleSpikeEndcap );

  clusterAlgo_.setNNeighbours( nNeighbours );

  // setup the position calculation (only affects ECAL position correction)
  std::string poscalctype = PositionCalcType__PFPositionCalc;
  if( iConfig.existsAs<std::string>("PositionCalcType") ) {
    poscalctype = iConfig.getParameter<std::string>("PositionCalcType");
  }
  if( poscalctype == PositionCalcType__EGPositionCalc ) {
    clusterAlgo_.setPositionCalcType(PFClusterAlgo::EGPositionCalc);    
    edm::ParameterSet pc_config = 
      iConfig.getParameterSet("PositionCalcConfig");
    clusterAlgo_.setEGammaPosCalc(pc_config);
  } else if( poscalctype == PositionCalcType__EGPositionFormula) {
    clusterAlgo_.setPositionCalcType(PFClusterAlgo::EGPositionFormula);
    edm::ParameterSet pc_config = 
      iConfig.getParameterSet("PositionCalcConfig");
    double w0 = pc_config.getParameter<double>("W0");
    clusterAlgo_.setPosCalcW0(w0);
  } else if( poscalctype == PositionCalcType__PFPositionCalc) {
    clusterAlgo_.setPositionCalcType(PFClusterAlgo::PFPositionCalc);
  } else {
    throw cms::Exception("InvalidClusteringType")
      << "You have not chosen a valid position calculation type,"
      << " please choose from \""
      << PositionCalcType__EGPositionCalc << "\", \""
      << PositionCalcType__EGPositionFormula << "\", or \""
      << PositionCalcType__PFPositionCalc << "\"!";
  }

  // p1 set to the minimum rechit threshold:
  double posCalcP1 = threshBarrel<threshEndcap ? threshBarrel:threshEndcap;
  clusterAlgo_.setPosCalcP1( posCalcP1 );
  clusterAlgo_.setPosCalcNCrystal( posCalcNCrystal );
  clusterAlgo_.setShowerSigma( showerSigma );

  clusterAlgo_.setUseCornerCells( useCornerCells  );
  clusterAlgo_.setCleanRBXandHPDs( cleanRBXandHPDs);

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


  geom = NULL;
  // access to the collections of rechits from the various detectors:

  
  inputTagPFRecHits_ = 
    iConfig.getParameter<InputTag>("PFRecHits");
  //---ab

  //inputTagClusterCollectionName_ =  iConfig.getParameter<string>("PFClusterCollectionName");    
 
  // produces<reco::PFClusterCollection>(inputTagClusterCollectionName_);
   produces<reco::PFClusterCollection>();
   produces<reco::PFRecHitCollection>("Cleaned");

    //---ab
}



PFClusterProducer::~PFClusterProducer() {}


void PFClusterProducer::beginLuminosityBlock(edm::LuminosityBlock const& iL, 
					     edm::EventSetup const& iE) {
  const CaloGeometryRecord& temp = iE.get<CaloGeometryRecord>();
  if( geom == NULL || (geom->cacheIdentifier() != temp.cacheIdentifier()) ) {
    geom = &temp;    
    edm::ESHandle<CaloGeometry> cgeom;
    geom->get(cgeom);
    clusterAlgo_.setEBGeom(cgeom->getSubdetectorGeometry(DetId::Ecal,
							 EcalBarrel));
    clusterAlgo_.setEEGeom(cgeom->getSubdetectorGeometry(DetId::Ecal,
							 EcalEndcap));
    clusterAlgo_.setPreshowerGeom(cgeom->getSubdetectorGeometry(DetId::Ecal,
							       EcalPreshower));
  }
}

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
  auto_ptr< vector<reco::PFCluster> > outClusters( clusterAlgo_.clusters() ); 
  auto_ptr< vector<reco::PFRecHit> > recHitsCleaned ( clusterAlgo_.rechitsCleaned() ); 
  iEvent.put( outClusters );    
  iEvent.put( recHitsCleaned, "Cleaned" );    

}
  



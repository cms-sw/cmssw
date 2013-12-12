#include "RecoParticleFlow/PFClusterProducer/plugins/PFClusterProducer.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"

#include <memory>

#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterAlgo.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"

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
#include "TVector2.h"


using namespace std;
using namespace edm;

namespace {
  const std::string PositionCalcType__EGPositionCalc("EGPositionCalc");
  const std::string PositionCalcType__EGPositionFormula("EGPositionFormula");
  const std::string PositionCalcType__PFPositionCalc("PFPositionCalc");
  typedef reco::PFCluster::EEtoPSAssociation::value_type EEPSPair;
  bool sortByKey(const EEPSPair& a, const EEPSPair& b) {
    return a.first < b.first;
  } 
  double testPreshowerDistance(const edm::Ptr<reco::PFCluster>& eeclus,
			       const edm::Ptr<reco::PFCluster>& psclus) {
    if( psclus.isNull() ) return -1.0;
    /* 
    // commented out since PFCluster::layer() uses a lot of CPU
    // and since 
    if( PFLayer::ECAL_ENDCAP != eeclus->layer() ) return -1.0;
    if( PFLayer::PS1 != psclus->layer() &&
	PFLayer::PS2 != psclus->layer()    ) {
      throw cms::Exception("testPreshowerDistance")
	<< "The second argument passed to this function was "
	<< "not a preshower cluster!" << std::endl;
    } 
    */
    const reco::PFCluster::REPPoint& pspos = psclus->positionREP();
    const reco::PFCluster::REPPoint& eepos = eeclus->positionREP();
    // lazy continue based on geometry
    if( eeclus->z()*psclus->z() < 0 ) return -1.0;
    const double dphi= std::abs(TVector2::Phi_mpi_pi(eepos.phi() - 
						     pspos.phi()));
    if( dphi > 0.6 ) return -1.0;    
    const double deta= std::abs(eepos.eta() - pspos.eta());    
    if( deta > 0.3 ) return -1.0; 
    return LinkByRecHit::testECALAndPSByRecHit(*eeclus,*psclus,false);
  }
}

PFClusterProducer::PFClusterProducer(const edm::ParameterSet& iConfig)
{
    
  verbose_ = 
    iConfig.getUntrackedParameter<bool>("verbose",false);
  
  pfEnergyCalibration_ = 
    std::shared_ptr<PFEnergyCalibration>(new PFEnergyCalibration());

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
  produces_eeps = iConfig.existsAs<InputTag>("PFClustersPS");
  if( produces_eeps ) {    
    inputTagPFClustersPS_ = 
      iConfig.getParameter<InputTag>("PFClustersPS");
    threshPFClusterES_ = iConfig.getParameter<double>("thresh_Preshower");
    applyCrackCorrections_ = 
      iConfig.getParameter<bool>("applyCrackCorrections");
    if (inputTagPFClustersPS_.label().empty()) {
      produces_eeps = false;
    } else {
      produces<reco::PFCluster::EEtoPSAssociation>();
    }
  }

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
  edm::Handle< edm::View<reco::PFCluster> > psclustersHandle;
  
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
  
  std::auto_ptr< std::vector<reco::PFCluster> > clusters = 
    clusterAlgo_.clusters();

  if( produces_eeps ) {
    iEvent.getByLabel( inputTagPFClustersPS_, psclustersHandle );
    // associate psclusters to ecal rechits
    std::auto_ptr<reco::PFCluster::EEtoPSAssociation> 
      outEEPS( new reco::PFCluster::EEtoPSAssociation );
    // make the association map of ECAL clusters to preshower clusters  
    edm::PtrVector<reco::PFCluster> clusterPtrsPS = 
      psclustersHandle->ptrVector();
    double dist = -1.0, min_dist = -1.0;
    // match PS clusters to EE clusters, minimum distance to EE is ensured
    // since the inner loop is over the EE clusters
    for( const auto& psclus : clusterPtrsPS ) {   
      if( psclus->energy() < threshPFClusterES_ ) continue;        
      switch( psclus->layer() ) { // just in case this isn't the ES...
      case PFLayer::PS1:
      case PFLayer::PS2:
	break;
      default:
	continue;
      }    
      edm::Ptr<reco::PFCluster> eematch,eeclus;
      dist = min_dist = -1.0; // reset
      for( size_t ic = 0; ic < clusters->size(); ++ic ) {
	eeclus = edm::Ptr<reco::PFCluster>(clusters.get(),ic);
	if( eeclus->layer() != PFLayer::ECAL_ENDCAP ) continue;	
	dist = testPreshowerDistance(eeclus,psclus);      
	if( dist == -1.0 || (min_dist != -1.0 && dist > min_dist) ) continue;
	if( dist < min_dist || min_dist == -1.0 ) {
	  eematch = eeclus;
	  min_dist = dist;
	}
      } // loop on EE clusters      
      if( eematch.isNonnull() ) {
	outEEPS->push_back(std::make_pair(eematch.key(),psclus));
      }
    } // loop on PS clusters    
    //sort the ps association
    std::sort(outEEPS->begin(),outEEPS->end(),sortByKey);
    
    std::vector<double> ps1_energies,ps2_energies;
    double ePS1, ePS2;
    for( size_t ic = 0 ; ic < clusters->size(); ++ic ) {
      ps1_energies.clear();
      ps2_energies.clear();
      ePS1 = ePS2 = 0;
      auto ee_key_val = std::make_pair(ic,edm::Ptr<reco::PFCluster>());
      const auto clustops = std::equal_range(outEEPS->begin(),
					     outEEPS->end(),
					     ee_key_val,
					     sortByKey);
      for( auto i_ps = clustops.first; i_ps != clustops.second; ++i_ps) {
	edm::Ptr<reco::PFCluster> psclus(i_ps->second);
	switch( psclus->layer() ) {
	case PFLayer::PS1:
	  ps1_energies.push_back(psclus->energy());
	  break;
	case PFLayer::PS2:
	  ps2_energies.push_back(psclus->energy());
	  break;
	default:
	  break;
	}
      }
      const double eCorr= 
	pfEnergyCalibration_->energyEm(clusters->at(ic),
				       ps1_energies,ps2_energies,
				       ePS1,ePS2,
				       applyCrackCorrections_);
      clusters->at(ic).setCorrectedEnergy(eCorr);
    }

    iEvent.put(outEEPS);
  }

  // get clusters out of the clustering algorithm 
  // and put them in the event. There is no copy.
  //auto_ptr< vector<reco::PFCluster> > outClusters( clusters ); 
  auto_ptr< vector<reco::PFRecHit> > recHitsCleaned ( clusterAlgo_.rechitsCleaned() ); 
  edm::OrphanHandle<reco::PFClusterCollection> outHandle = 
    iEvent.put( clusters );    
  iEvent.put( recHitsCleaned, "Cleaned" ); 
}
  



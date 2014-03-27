#include "RecoEcal/EgammaClusterAlgos/interface/PFECALSuperClusterAlgo.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFClusterWidthAlgo.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "RecoEcal/EgammaCoreTools/interface/Mustache.h"
#include "Math/GenVector/VectorUtil.h"
#include "TFile.h"
#include "TH2F.h"
#include "TROOT.h"
#include "TMath.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <stdexcept>
#include <string>
#include <sstream>
#include <cmath>

using namespace std;
namespace MK = reco::MustacheKernel;

namespace {
  typedef edm::View<reco::PFCluster> PFClusterView;
  typedef edm::Ptr<reco::PFCluster> PFClusterPtr;
  typedef edm::PtrVector<reco::PFCluster> PFClusterPtrVector;
  typedef PFECALSuperClusterAlgo::CalibratedClusterPtr CalibClusterPtr;
  typedef PFECALSuperClusterAlgo::CalibratedClusterPtrVector CalibClusterPtrVector;
  typedef std::pair<reco::CaloClusterPtr::key_type,reco::CaloClusterPtr> EEPSPair;
  typedef std::binary_function<const CalibClusterPtr&,
			       const CalibClusterPtr&, 
			       bool> ClusBinaryFunction;
  
  typedef std::unary_function<const CalibClusterPtr&, 
			      bool> ClusUnaryFunction;  

  bool sortByKey(const EEPSPair& a, const EEPSPair& b) {
    return a.first < b.first;
  } 

  double getPFClusterEnergy(const PFClusterPtr& p) {
    return p->energy();
  }  

  inline double ptFast( const double energy, 
			const math::XYZPoint& position,
			const math::XYZPoint& origin ) {
    const auto v = position - origin;
    return energy*std::sqrt(v.perp2()/v.mag2());
  }

  struct SumPSEnergy : public std::binary_function<double,
						   const PFClusterPtr&,
						   double> {
    PFLayer::Layer _thelayer;
    SumPSEnergy(PFLayer::Layer layer) : _thelayer(layer) {}
    double operator()(double a,
		      const PFClusterPtr& b) {
      return a + (_thelayer == b->layer())*b->energy();
    }
  };

  struct GreaterByE : public ClusBinaryFunction {
    bool operator()(const CalibClusterPtr& x, 
		    const CalibClusterPtr& y) { 
      return x->energy() > y->energy() ; 
    }
  };

  struct GreaterByEt : public ClusBinaryFunction {
    bool operator()(const CalibClusterPtr& x, 
		    const CalibClusterPtr& y) {
      const math::XYZPoint zero(0,0,0);
      const double xpt = ptFast(x->energy(),x->the_ptr()->position(),zero);
      const double ypt = ptFast(y->energy(),y->the_ptr()->position(),zero);
      return xpt > ypt;
    }
  };

  struct IsASeed : public ClusUnaryFunction {
    const double threshold;
    const bool cutET;
    IsASeed(double thresh, bool useETcut = false) : 
      threshold(thresh), cutET(useETcut) {}
    bool operator()(const CalibClusterPtr& x) {
      const math::XYZPoint zero(0,0,0);
      double e_or_et = x->energy();
      if( cutET )  e_or_et = ptFast(e_or_et,x->the_ptr()->position(),zero);
      return e_or_et > threshold; 
    }
  };

  struct IsLinkedByRecHit : public ClusUnaryFunction {
    const CalibClusterPtr the_seed;
    const double _threshold, _majority;
    const double _maxSatelliteDEta, _maxSatelliteDPhi;
	double x_rechits_tot=0;
    double x_rechits_match=0;
    IsLinkedByRecHit(const CalibClusterPtr& s, const double threshold,
		     const double majority, const double maxDEta,
		     const double maxDPhi) : 
      the_seed(s),_threshold(threshold),_majority(majority), 
      _maxSatelliteDEta(maxDEta), _maxSatelliteDPhi(maxDPhi) {}
    bool operator()(const CalibClusterPtr& x) {      
      if( the_seed->energy_nocalib() < _threshold ) return false; 
      const double dEta = std::abs(the_seed->eta()-x->eta());
      const double dPhi = 
	std::abs(TVector2::Phi_mpi_pi(the_seed->phi() - x->phi())); 
      if( _maxSatelliteDEta < dEta || _maxSatelliteDPhi < dPhi) return false;
      // now see if the clusters overlap in rechits
      const auto& seedHitsAndFractions = 
	the_seed->the_ptr()->hitsAndFractions();
      const auto& xHitsAndFractions = 
	x->the_ptr()->hitsAndFractions();      
      x_rechits_tot   = xHitsAndFractions.size();
      x_rechits_match = 0.0;      
      for( const std::pair<DetId, float>& seedHit : seedHitsAndFractions ) {
	for( const std::pair<DetId, float>& xHit : xHitsAndFractions ) {
	  if( seedHit.first == xHit.first ) {	    
	    x_rechits_match += 1.0;
	  }
	}	
      }      
      return x_rechits_match/x_rechits_tot > _majority;
    }
  };

  struct IsClustered : public ClusUnaryFunction {
    const CalibClusterPtr the_seed;    
    PFECALSuperClusterAlgo::clustering_type _type;
    bool dynamic_dphi;
    double etawidthSuperCluster_ = .0 , phiwidthSuperCluster_ = .0;
    IsClustered(const CalibClusterPtr s, 
		PFECALSuperClusterAlgo::clustering_type ct,
		const bool dyn_dphi) : 
      the_seed(s), _type(ct), dynamic_dphi(dyn_dphi) {}
    bool operator()(const CalibClusterPtr& x) { 
      const double dphi = 
	std::abs(TVector2::Phi_mpi_pi(the_seed->phi() - x->phi()));        
      const bool passes_dphi = 
	( (!dynamic_dphi && dphi < phiwidthSuperCluster_ ) || 
	  (dynamic_dphi && MK::inDynamicDPhiWindow(the_seed->eta(),
						   the_seed->phi(),
						   x->energy_nocalib(),
						   x->eta(),
						   x->phi()) ) );

      switch( _type ) {
      case PFECALSuperClusterAlgo::kBOX:
	return ( std::abs(the_seed->eta()-x->eta())<etawidthSuperCluster_ && 
		 passes_dphi   );
	break;
      case PFECALSuperClusterAlgo::kMustache:
	return ( passes_dphi && 
		 MK::inMustache(the_seed->eta(), 
				the_seed->phi(),
				x->energy_nocalib(),
				x->eta(),
				x->phi()            ));
	break;
      default: 
	return false;
      }
      return false;
    }
  };
}

PFECALSuperClusterAlgo::PFECALSuperClusterAlgo() : beamSpot_(0) { }

void PFECALSuperClusterAlgo::
setPFClusterCalibration(const std::shared_ptr<PFEnergyCalibration>& calib) {
  _pfEnergyCalibration = calib;
}

void PFECALSuperClusterAlgo::setTokens(const edm::ParameterSet &iConfig, edm::ConsumesCollector &&cc) {
  
  inputTagPFClusters_ = 
    cc.consumes<edm::View<reco::PFCluster> >(iConfig.getParameter<edm::InputTag>("PFClusters"));
  inputTagPFClustersES_ = 
    cc.consumes<reco::PFCluster::EEtoPSAssociation>(iConfig.getParameter<edm::InputTag>("ESAssociation"));  
  inputTagBeamSpot_ = 
    cc.consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("BeamSpot"));    
    
  if (useRegression_) {
    const edm::ParameterSet &regconf = iConfig.getParameter<edm::ParameterSet>("regressionConfig");
    
    regr_.reset(new PFSCRegressionCalc(regconf));
    regr_->varCalc()->setTokens(regconf,std::move(cc));  
  }
  
}

void PFECALSuperClusterAlgo::update(const edm::EventSetup& setup) {
 
  if (useRegression_) {
    regr_->update(setup);
  }
  
}



void PFECALSuperClusterAlgo::
loadAndSortPFClusters(const edm::Event &iEvent) { 
  
  //load input collections
  //Load the pfcluster collections
  edm::Handle<edm::View<reco::PFCluster> > pfclustersHandle;
  iEvent.getByToken( inputTagPFClusters_, pfclustersHandle );  

  edm::Handle<reco::PFCluster::EEtoPSAssociation > psAssociationHandle;
  iEvent.getByToken( inputTagPFClustersES_,  psAssociationHandle);  
  
  const PFClusterView& clusters = *pfclustersHandle.product();
  const reco::PFCluster::EEtoPSAssociation& psclusters = *psAssociationHandle.product();
  
  //load BeamSpot
  edm::Handle<reco::BeamSpot> bsHandle;
  iEvent.getByToken( inputTagBeamSpot_, bsHandle);
  beamSpot_ = bsHandle.product();
  
  //initialize regression for this event
  if (useRegression_) {
    regr_->varCalc()->setEvent(iEvent);
  }  
  
  // reset the system for running
  superClustersEB_.reset(new reco::SuperClusterCollection);
  _clustersEB.clear();
  superClustersEE_.reset(new reco::SuperClusterCollection);    
  _clustersEE.clear();  
  EEtoPS_ = &psclusters;

  auto clusterPtrs = clusters.ptrVector(); 
  //Select PF clusters available for the clustering
  for ( auto& cluster : clusterPtrs ){
    LogDebug("PFClustering") 
      << "Loading PFCluster i="<<cluster.key()
      <<" energy="<<cluster->energy()<<std::endl;
        
    CalibratedClusterPtr calib_cluster(new CalibratedPFCluster(cluster));
    switch( cluster->layer() ) {
    case PFLayer::ECAL_BARREL:
      if( calib_cluster->energy() > threshPFClusterBarrel_ ) {
	_clustersEB.push_back(calib_cluster);	
      }
      break;
    case PFLayer::ECAL_ENDCAP:
      if( calib_cluster->energy() > threshPFClusterEndcap_ ) {
	_clustersEE.push_back(calib_cluster);
      }
      break;
    default:
      break;
    }
  }
  // sort full cluster collections by their calibrated energy
  // this will put all the seeds first by construction
  GreaterByEt greater;
  std::sort(_clustersEB.begin(), _clustersEB.end(), greater);
  std::sort(_clustersEE.begin(), _clustersEE.end(), greater);  
  
}

void PFECALSuperClusterAlgo::run() {  
  // clusterize the EB
  buildAllSuperClusters(_clustersEB, threshPFClusterSeedBarrel_);
  // clusterize the EE
  buildAllSuperClusters(_clustersEE, threshPFClusterSeedEndcap_);
}

void PFECALSuperClusterAlgo::
buildAllSuperClusters(CalibClusterPtrVector& clusters,
		      double seedthresh) {
  IsASeed seedable(seedthresh,threshIsET_);
  // make sure only seeds appear at the front of the list of clusters
  std::stable_partition(clusters.begin(),clusters.end(),seedable);
  // in each iteration we are working on a list that is already sorted
  // in the cluster energy and remains so through each iteration
  // NB: since clusters is sorted in loadClusters any_of has O(1)
  //     timing until you run out of seeds!  
  while( std::any_of(clusters.cbegin(), clusters.cend(), seedable) ) {    
    buildSuperCluster(clusters.front(),clusters);
  }
}

void PFECALSuperClusterAlgo::
buildSuperCluster(CalibClusterPtr& seed,
		  CalibClusterPtrVector& clusters) {
  IsClustered IsClusteredWithSeed(seed,_clustype,_useDynamicDPhi);
  IsLinkedByRecHit MatchesSeedByRecHit(seed,satelliteThreshold_,
				       fractionForMajority_,0.1,0.2);
  bool isEE = false;
  SumPSEnergy sumps1(PFLayer::PS1), sumps2(PFLayer::PS2);  
  switch( seed->the_ptr()->layer() ) {
  case PFLayer::ECAL_BARREL:
    IsClusteredWithSeed.phiwidthSuperCluster_ = phiwidthSuperClusterBarrel_;
    IsClusteredWithSeed.etawidthSuperCluster_ = etawidthSuperClusterBarrel_;
    edm::LogInfo("PFClustering") << "Building SC number "  
				 << superClustersEB_->size() + 1
				 << " in the ECAL barrel!";
    break;
  case PFLayer::ECAL_ENDCAP:    
    IsClusteredWithSeed.phiwidthSuperCluster_ = phiwidthSuperClusterEndcap_; 
    IsClusteredWithSeed.etawidthSuperCluster_ = etawidthSuperClusterEndcap_;
    edm::LogInfo("PFClustering") << "Building SC number "  
				 << superClustersEE_->size() + 1
				 << " in the ECAL endcap!" << std::endl;
    isEE = true;
    break;
  default:
    break;
  }
  
  // this function shuffles the list of clusters into a list
  // where all clustered sub-clusters are at the front 
  // and returns a pointer to the first unclustered cluster.
  // The relative ordering of clusters is preserved 
  // (i.e. both resulting sub-lists are sorted by energy).
  auto not_clustered = std::stable_partition(clusters.begin(),clusters.end(),
					     IsClusteredWithSeed);
  // satellite cluster merging
  // it was found that large clusters can split!
  if( doSatelliteClusterMerge_ ) {    
    not_clustered = std::stable_partition(not_clustered,clusters.end(),
					  MatchesSeedByRecHit);
  }

  if(verbose_) {
    edm::LogInfo("PFClustering") << "Dumping cluster detail";
    edm::LogVerbatim("PFClustering")
      << "\tPassed seed: e = " << seed->energy_nocalib() 
      << " eta = " << seed->eta() << " phi = " << seed->phi() 
      << std::endl;  
    for( auto clus = clusters.cbegin(); clus != not_clustered; ++clus ) {
      edm::LogVerbatim("PFClustering") 
	<< "\t\tClustered cluster: e = " << (*clus)->energy_nocalib() 
	<< " eta = " << (*clus)->eta() << " phi = " << (*clus)->phi() 
	<< std::endl;
    }
    for( auto clus = not_clustered; clus != clusters.end(); ++clus ) {
      edm::LogVerbatim("PFClustering") 
	<< "\tNon-Clustered cluster: e = " << (*clus)->energy_nocalib() 
	<< " eta = " << (*clus)->eta() << " phi = " << (*clus)->phi() 
	<< std::endl;
    }    
  }
  // move the clustered clusters out of available cluster list
  // and into a temporary vector for building the SC  
  CalibratedClusterPtrVector clustered(clusters.begin(),not_clustered);
  clusters.erase(clusters.begin(),not_clustered);    
  // need the vector of raw pointers for a PF width class
  std::vector<const reco::PFCluster*> bare_ptrs;
  // calculate necessary parameters and build the SC
  double posX(0), posY(0), posZ(0),
    corrSCEnergy(0), corrPS1Energy(0), corrPS2Energy(0), 
    ePS1(0), ePS2(0), energyweight(0), energyweighttot(0);
  std::vector<double> ps1_energies, ps2_energies;
  for( auto& clus : clustered ) {    
    ePS1 = ePS2 = 0;
    energyweight = clus->energy_nocalib();
    bare_ptrs.push_back(clus->the_ptr().get());
    // update EE calibrated super cluster energies
    if( isEE ) {
      ps1_energies.clear();
      ps2_energies.clear();
      auto ee_key_val = 
	std::make_pair(clus->the_ptr().key(),edm::Ptr<reco::PFCluster>());
      const auto clustops = std::equal_range(EEtoPS_->begin(),
					     EEtoPS_->end(),
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
      _pfEnergyCalibration->energyEm(*(clus->the_ptr()),
				     ps1_energies,ps2_energies,
				     ePS1,ePS2,
				     applyCrackCorrections_);
    }
    
    switch( _eweight ) {
    case kRaw: // energyweight is initialized to raw cluster energy
      break;
    case kCalibratedNoPS:
      energyweight = clus->energy() - ePS1 - ePS2;
      break;
    case kCalibratedTotal:
      energyweight = clus->energy();
      break;
    default:
      break;
    }    
    const math::XYZPoint& cluspos = clus->the_ptr()->position();
    posX += energyweight * cluspos.X();
    posY += energyweight * cluspos.Y();
    posZ += energyweight * cluspos.Z();    

    energyweighttot += energyweight;
    corrSCEnergy    += clus->energy();   
    corrPS1Energy   += ePS1;
    corrPS2Energy   += ePS2;
  }
  posX /= energyweighttot;
  posY /= energyweighttot;
  posZ /= energyweighttot;    
  
  // now build the supercluster
  reco::SuperCluster new_sc(corrSCEnergy,math::XYZPoint(posX,posY,posZ));   
  new_sc.setCorrectedEnergy(corrSCEnergy);
  new_sc.setSeed(clustered.front()->the_ptr());
  new_sc.setPreshowerEnergy(corrPS1Energy+corrPS2Energy);
  new_sc.setPreshowerEnergyPlane1(corrPS1Energy);
  new_sc.setPreshowerEnergyPlane2(corrPS2Energy);
  for( const auto& clus : clustered ) {
    new_sc.addCluster(clus->the_ptr());
    
    auto& hits_and_fractions = clus->the_ptr()->hitsAndFractions();
    for( auto& hit_and_fraction : hits_and_fractions ) {
      new_sc.addHitAndFraction(hit_and_fraction.first,hit_and_fraction.second);
    }
    if( isEE ) {
      auto ee_key_val = 
	std::make_pair(clus->the_ptr().key(),edm::Ptr<reco::PFCluster>());
      const auto clustops = std::equal_range(EEtoPS_->begin(),
					     EEtoPS_->end(),
					     ee_key_val,
					     sortByKey);
      // EE rechits should be uniquely matched to sets of pre-shower
      // clusters at this point, so we throw an exception if otherwise
      // now wrapped in EDM debug flags
      for( auto i_ps = clustops.first; i_ps != clustops.second; ++i_ps) {
	edm::Ptr<reco::PFCluster> psclus(i_ps->second);
#ifdef EDM_ML_DEBUG
	auto found_pscluster = std::find(new_sc.preshowerClustersBegin(),
					 new_sc.preshowerClustersEnd(),
					 i_ps->second);
	if( found_pscluster == new_sc.preshowerClustersEnd() ) {
#endif	  
	  new_sc.addPreshowerCluster(psclus);	  	  
#ifdef EDM_ML_DEBUG
	} else {
	  throw cms::Exception("PFECALSuperClusterAlgo::buildSuperCluster")
	    << "Found a PS cluster matched to more than one EE cluster!" 
	    << std::endl << std::hex << psclus.get() << " == " 
	    << found_pscluster->get() << std::dec << std::endl;
	}
#endif
      }
    }
  }  
  
  // calculate linearly weighted cluster widths
  PFClusterWidthAlgo pfwidth(bare_ptrs);
  new_sc.setEtaWidth(pfwidth.pflowEtaWidth());
  new_sc.setPhiWidth(pfwidth.pflowPhiWidth());
  
  // cache the value of the raw energy  
  new_sc.rawEnergy();

  //apply regression energy corrections
  if( useRegression_ ) {    
    double cor = regr_->getCorrection(new_sc);
    new_sc.setEnergy(cor*new_sc.correctedEnergy());
  }
  
  // save the super cluster to the appropriate list (if it passes the final
  // Et threshold)
  //Note that Et is computed here with respect to the beamspot position
  //in order to be consistent with the cut applied in the
  //ElectronSeedProducer
  double scEtBS = 
    ptFast(new_sc.energy(),new_sc.position(),beamSpot_->position());

  if ( scEtBS > threshSuperClusterEt_ ) {
    switch( seed->the_ptr()->layer() ) {
    case PFLayer::ECAL_BARREL:
      superClustersEB_->push_back(new_sc);
      break;
    case PFLayer::ECAL_ENDCAP:    
      superClustersEE_->push_back(new_sc);    
      break;
    default:
      break;
    }
  }
}

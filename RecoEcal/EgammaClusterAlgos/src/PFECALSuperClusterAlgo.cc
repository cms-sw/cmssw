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

  typedef std::binary_function<const CalibClusterPtr&,
			       const CalibClusterPtr&, 
			       bool> ClusBinaryFunction;
  
  typedef std::unary_function<const CalibClusterPtr&, 
			      bool> ClusUnaryFunction;  

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
      return x->energy()/std::cosh(x->eta()) > y->energy()/std::cosh(y->eta());
    }
  };

  struct IsASeed : public ClusUnaryFunction {
    double threshold;
    IsASeed(double thresh) : threshold(thresh) {}
    bool operator()(const CalibClusterPtr& x) { 
      return x->energy() > threshold; 
    }
  };

  struct IsLinkedByRecHit : public ClusUnaryFunction {
    const CalibClusterPtr the_seed;
    const double _threshold, _majority;
    const double _maxSatelliteDEta, _maxSatelliteDPhi;
    double x_rechits_tot, x_rechits_match;
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
    double etawidthSuperCluster_, phiwidthSuperCluster_;
    IsClustered(const CalibClusterPtr s, 
		PFECALSuperClusterAlgo::clustering_type ct,
		const bool dyn_dphi) : 
      the_seed(s), _type(ct), dynamic_dphi(dyn_dphi) {}
    bool operator()(const CalibClusterPtr& x) { 
      const double dphi = 
	std::abs(TVector2::Phi_mpi_pi(the_seed->phi() - x->phi()));  
      const bool isEB = ( PFLayer::ECAL_BARREL == x->the_ptr()->layer() );
      const bool passes_dphi = 
	( (!dynamic_dphi && dphi < phiwidthSuperCluster_ ) || 
	  (dynamic_dphi && MK::inDynamicDPhiWindow(isEB,the_seed->phi(),
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

  double testPreshowerDistance(const edm::Ptr<reco::PFCluster>& eeclus,
			       const edm::Ptr<reco::PFCluster>& psclus) {
    if( psclus.isNull() || eeclus.isNull() ) return -1.0;
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

PFECALSuperClusterAlgo::PFECALSuperClusterAlgo() { }

void PFECALSuperClusterAlgo::
setPFClusterCalibration(const std::shared_ptr<PFEnergyCalibration>& calib) {
  _pfEnergyCalibration = calib;
}

void PFECALSuperClusterAlgo::
loadAndSortPFClusters(const PFClusterView& clusters,
		      const PFClusterView& psclusters) { 
  // reset the system for running
  superClustersEB_.reset(new reco::SuperClusterCollection);
  _clustersEB.clear();
  superClustersEE_.reset(new reco::SuperClusterCollection);  
  _clustersEE.clear();
  _psclustersforee.clear();
  
  auto clusterPtrs = clusters.ptrVector(); 
  //Select PF clusters available for the clustering
  for ( auto& cluster : clusterPtrs ){
    LogDebug("PFClustering") 
      << "Loading PFCluster i="<<cluster.key()
      <<" energy="<<cluster->energy()<<std::endl;
    
    double Ecorr = _pfEnergyCalibration->energyEm(*cluster,
						  0.0,0.0,
						  applyCrackCorrections_);
    CalibratedClusterPtr calib_cluster(new CalibratedPFCluster(cluster,Ecorr));
    switch( cluster->layer() ) {
    case PFLayer::ECAL_BARREL:
      if( calib_cluster->energy() > threshPFClusterBarrel_ ) {
	_clustersEB.push_back(calib_cluster);	
      }
      break;
    case PFLayer::ECAL_ENDCAP:
      if( calib_cluster->energy() > threshPFClusterEndcap_ ) {
	_clustersEE.push_back(calib_cluster);
	_psclustersforee.emplace(calib_cluster->the_ptr(),
				 edm::PtrVector<reco::PFCluster>());
      }
      break;
    default:
      break;
    }
  }
  // make the association map of ECAL clusters to preshower clusters  
  edm::PtrVector<reco::PFCluster> clusterPtrsPS = psclusters.ptrVector();
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
    edm::Ptr<reco::PFCluster> eematch;
    dist = min_dist = -1.0; // reset
    for( const auto& eeclus : _clustersEE ) {
      dist = testPreshowerDistance(eeclus->the_ptr(),psclus);      
      if( dist == -1.0 || (min_dist != -1.0 && dist > min_dist) ) continue;
      if( dist < min_dist || min_dist == -1.0 ) {
	eematch = eeclus->the_ptr();
	min_dist = dist;
      }
    } // loop on EE clusters      
    if( eematch.isNonnull() ) _psclustersforee[eematch].push_back(psclus);
  } // loop on PS clusters

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
  IsASeed seedable(seedthresh);
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
    rawSCEnergy(0), corrSCEnergy(0), clusterCorrEE(0), 
    PS1_clus_sum(0), PS2_clus_sum(0);  
  for( auto& clus : clustered ) {
    bare_ptrs.push_back(clus->the_ptr().get());
      
    const double cluseraw = clus->energy_nocalib();
    const math::XYZPoint& cluspos = clus->the_ptr()->position();
    posX += cluseraw * cluspos.X();
    posY += cluseraw * cluspos.Y();
    posZ += cluseraw * cluspos.Z();
    // update EE calibrated super cluster energies
    if( isEE ) {
      const auto& psclusters = _psclustersforee[clus->the_ptr()];
      PS1_clus_sum = std::accumulate(psclusters.begin(),psclusters.end(),
				     0.0,sumps1);
      PS2_clus_sum = std::accumulate(psclusters.begin(),psclusters.end(),
				     0.0,sumps2);
      clusterCorrEE = 
	_pfEnergyCalibration->energyEm(*(clus->the_ptr()),
				       PS1_clus_sum,PS2_clus_sum,
				       applyCrackCorrections_);
      clus->resetCalibratedEnergy(clusterCorrEE);
    }

    rawSCEnergy  += cluseraw;
    corrSCEnergy += clus->energy();    
  }
  posX /= rawSCEnergy;
  posY /= rawSCEnergy;
  posZ /= rawSCEnergy;    
  
  // now build the supercluster
  reco::SuperCluster new_sc(corrSCEnergy,math::XYZPoint(posX,posY,posZ)); 
  double ps1_energy(0.0), ps2_energy(0.0), ps_energy(0.0);
  new_sc.setSeed(clustered.front()->the_ptr());
  for( const auto& clus : clustered ) {
    new_sc.addCluster(clus->the_ptr());
    auto& hits_and_fractions = clus->the_ptr()->hitsAndFractions();
    for( auto& hit_and_fraction : hits_and_fractions ) {
      new_sc.addHitAndFraction(hit_and_fraction.first,hit_and_fraction.second);
    }
    const auto& cluspsassociation = _psclustersforee[clus->the_ptr()];     
    // EE rechits should be uniquely matched to sets of pre-shower
    // clusters at this point, so we throw an exception if otherwise
    for( const auto& psclus : cluspsassociation ) {
      auto found_pscluster = std::find(new_sc.preshowerClustersBegin(),
				       new_sc.preshowerClustersEnd(),
				       reco::CaloClusterPtr(psclus));
      if( found_pscluster == new_sc.preshowerClustersEnd() ) {
	const double psenergy = psclus->energy();
	new_sc.addPreshowerCluster(psclus);
	ps1_energy += (PFLayer::PS1 == psclus->layer())*psenergy;
	ps2_energy += (PFLayer::PS2 == psclus->layer())*psenergy;
	ps_energy  += psenergy;
      } else {
	throw cms::Exception("PFECALSuperClusterAlgo::buildSuperCluster")
	  << "Found a PS cluster matched to more than one EE cluster!" 
	  << std::endl << std::hex << psclus.get() << " == " 
	  << found_pscluster->get() << std::dec << std::endl;
      }
    }
  }
  new_sc.setPreshowerEnergy(ps_energy); 
  new_sc.setPreshowerEnergyPlane1(ps1_energy);
  new_sc.setPreshowerEnergyPlane2(ps2_energy);
  
  // calculate linearly weighted cluster widths
  PFClusterWidthAlgo pfwidth(bare_ptrs);
  new_sc.setEtaWidth(pfwidth.pflowEtaWidth());
  new_sc.setPhiWidth(pfwidth.pflowPhiWidth());
  
  // cache the value of the raw energy  
  new_sc.rawEnergy();

  // save the super cluster to the appropriate list
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

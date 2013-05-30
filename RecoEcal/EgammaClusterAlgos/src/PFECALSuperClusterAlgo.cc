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

using namespace std;

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
      return x->energy_nocalib() > y->energy_nocalib() ; 
    }
  };

  struct IsASeed : public ClusUnaryFunction {
    double threshold;
    IsASeed(double thresh) : threshold(thresh) {}
    bool operator()(const CalibClusterPtr& x) { 
      return x->energy() > threshold; 
    }
  };

  struct IsClustered : public ClusUnaryFunction {
    const CalibClusterPtr the_seed;
    PFECALSuperClusterAlgo::clustering_type _type;
    double etawidthSuperCluster_, phiwidthSuperCluster_;
    IsClustered(const CalibClusterPtr s, 
		PFECALSuperClusterAlgo::clustering_type ct) : 
      the_seed(s), _type(ct) {}
    bool operator()(const CalibClusterPtr& x) { 
      double dphi = 
	std::abs(TVector2::Phi_mpi_pi(the_seed->phi() - x->phi()));  
      
      switch( _type ) {
      case PFECALSuperClusterAlgo::kBOX:
	return ( std::abs(the_seed->eta()-x->eta())<etawidthSuperCluster_ && 
		 dphi < phiwidthSuperCluster_   );
	break;
      case PFECALSuperClusterAlgo::kMustache:
	return ( dphi < phiwidthSuperCluster_ &&
		 reco::MustacheKernel::inMustache(the_seed->eta(), 
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

PFECALSuperClusterAlgo::PFECALSuperClusterAlgo() {
}

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
  for ( auto cluster : clusterPtrs ){
    LogDebug("PFClustering") 
      << "Loading PFCluster i="<<cluster.key()
      <<" energy="<<cluster->energy()<<endl;
    
    double Ecorr = _pfEnergyCalibration->energyEm(*cluster,
						  0.0,0.0,
						  applyCrackCorrections_);
    CalibratedClusterPtr calib_cluster(new CalibratedPFCluster(cluster, 
							       Ecorr));
    switch( cluster->layer() ) {
    case PFLayer::ECAL_BARREL:
      if( calib_cluster->energy() > threshPFClusterBarrel_ ) {
	_clustersEB.push_back(calib_cluster);	
      }
      break;
    case PFLayer::ECAL_ENDCAP:
      if( calib_cluster->energy() > threshPFClusterEndcap_ ) {
	_clustersEE.push_back(calib_cluster);
	_psclustersforee.insert(std::make_pair(calib_cluster->the_ptr(),
					       PFClusterPtrVector()));      }
      break;
    default:
      break;
    }
  }
  // make the association map of ECAL clusters to preshower clusters  
  auto clusterPtrsPS = psclusters.ptrVector();
  double dist = -1.0, min_dist = -1.0;
  // match PS clusters to clusters from the EE (many to one relation)
  for( auto& psclus : clusterPtrsPS ) {
    if( psclus->energy() < threshPFClusterES_ ) continue;    
    switch( psclus->layer() ) { // just in case this isn't the ES...
    case PFLayer::PS1:
    case PFLayer::PS2:
      break;
    default:
      continue;
    }
    // now match the PS cluster to the closest EE cluster
    edm::Ptr<reco::PFCluster> eematch;
    min_dist = -1.0;
    for( auto& eeclus : _clustersEE ) {
      // lazy continue based on geometry
      double prod = eeclus->eta()*psclus->eta();
      double deta= std::abs(eeclus->eta() - psclus->eta());
      double dphi= std::abs(TVector2::Phi_mpi_pi(eeclus->phi() - 
						 psclus->phi()));
      if( prod < 0 || deta > 0.3 || dphi > 0.6 ) continue;
      // now we actually do the matching
      dist = LinkByRecHit::testECALAndPSByRecHit( *(eeclus->the_ptr()), 
						  *psclus, 
						  false);
      if( dist == -1.0 ) continue;
      if( dist < min_dist || min_dist == -1.0 ) eematch = eeclus->the_ptr();
      }
    if( eematch.isNull() ) continue; // lazy continue if no match found    
    _psclustersforee[eematch].push_back(psclus);
  }  

  // sort full cluster collections by their calibrated energy
  // this will put all the seeds first by construction
  GreaterByE greaterByE;
  std::sort(_clustersEB.begin(), _clustersEB.end(), greaterByE);
  std::sort(_clustersEE.begin(), _clustersEE.end(), greaterByE);  
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
  // in each iteration we are working on a list that is already sorted
  // in the cluster energy and remains so through each iteration
  // NB: since clusters is sorted in loadClusters any_of has O(1)
  //     timing until you run out of seeds!
  while( std::any_of(clusters.cbegin(), 
		     clusters.cend(),
		     seedable ) ) {    
    buildSuperCluster(clusters.front(),clusters);
  }
}

void PFECALSuperClusterAlgo::
buildSuperCluster(CalibClusterPtr& seed,
		  CalibClusterPtrVector& clusters) {
  IsClustered IsClusteredWithSeed(seed,_clustype);
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
  auto not_clustered = std::stable_partition(clusters.begin(), 
					     clusters.end(),
					     IsClusteredWithSeed);
  if(verbose_) {
    edm::LogInfo("PFClustering") << "Dumping cluster detail";
    edm::LogVerbatim("PFClustering")
      << "\tPassed seed: e = " << seed->energy_nocalib() 
      << " eta = " << seed->eta() 
      << " phi = " << seed->phi() <<std::endl;  
    for( auto clus = clusters.cbegin(); clus != not_clustered; ++clus ) {
      edm::LogVerbatim("PFClustering") 
	<< "\t\tClustered cluster: e = " 
	<< (*clus)->energy_nocalib() 
	<< " eta = " << (*clus)->eta() 
	<< " phi = " << (*clus)->phi() << std::endl;
    }
    for( auto clus = not_clustered; clus != clusters.end(); ++clus ) {
      edm::LogVerbatim("PFClustering") 
	<< "\tNon-Clustered cluster: e = " 
	<< (*clus)->energy_nocalib() 
	<< " eta = " << (*clus)->eta() 
	<< " phi = " << (*clus)->phi() << std::endl;
    }    
  }
  // move the clustered clusters out of available cluster list
  // and into a temporary vector for building the SC  
  CalibratedClusterPtrVector clustered;
  clustered.reserve(not_clustered - clusters.begin());
  clustered.insert(clustered.begin(),clusters.begin(),not_clustered);
  clusters.erase(clusters.begin(),not_clustered);    
  // need the vector of raw pointers for a PF width class
  std::vector<const reco::PFCluster*> bare_ptrs;
  // calculate necessary parameters and build the SC
  double posX(0), posY(0), posZ(0),
    rawSCEnergy(0), corrSCEnergy(0), clusterCorrEE(0), 
    PS1_clus_sum(0), PS2_clus_sum(0);
  edm::PtrVector<reco::PFCluster> psclusters;
  for( auto& clus : clustered ) {
    bare_ptrs.push_back(clus->the_ptr().get());
      
    posX += clus->energy_nocalib() * clus->the_ptr()->position().X();
    posY += clus->energy_nocalib() * clus->the_ptr()->position().Y();
    posZ += clus->energy_nocalib() * clus->the_ptr()->position().Z();
    // update EE calibrated super cluster energies
    if( isEE ) {
      psclusters = _psclustersforee.find(clus->the_ptr())->second;
      PS1_clus_sum = std::accumulate(psclusters.begin(),
				     psclusters.end(),
				     0.0,sumps1);
      PS2_clus_sum = std::accumulate(psclusters.begin(),
				     psclusters.end(),
				     0.0,sumps2);
      clusterCorrEE = 
	_pfEnergyCalibration->energyEm(*(clus->the_ptr()),
				       PS1_clus_sum,
				       PS2_clus_sum,
				       applyCrackCorrections_);
      clus->resetCalibratedEnergy(clusterCorrEE);
    }

    rawSCEnergy  += clus->energy_nocalib();
    corrSCEnergy += clus->energy();    
  }
  posX /= rawSCEnergy;
  posY /= rawSCEnergy;
  posZ /= rawSCEnergy;    
  
  // now build the supercluster
  reco::SuperCluster new_sc(corrSCEnergy,math::XYZPoint(posX,posY,posZ)); 
  double ps1_energy(0.0), ps2_energy(0.0), ps_energy(0.0);
  new_sc.setSeed(clustered.front()->the_ptr());
  for( auto& clus : clustered ) {
    new_sc.addCluster(clus->the_ptr());
    auto& hits_and_fractions = clus->the_ptr()->hitsAndFractions();
    for( auto& hit_and_fraction : hits_and_fractions ) {
      new_sc.addHitAndFraction(hit_and_fraction.first,hit_and_fraction.second);
    }
    auto cluspsassociation = _psclustersforee.find(clus->the_ptr());
    if( cluspsassociation != _psclustersforee.end() ) {    
      // since EE rechits can share PS rechits, we have to make
      // sure the PS rechit hasn't already been added
      for( auto& psclus : cluspsassociation->second ) {
	auto found_pscluster = std::find(new_sc.preshowerClustersBegin(),
					 new_sc.preshowerClustersEnd(),
					 reco::CaloClusterPtr(psclus));
	if( found_pscluster == new_sc.preshowerClustersEnd() ) {
	  ps1_energy += (PFLayer::PS1 == psclus->layer())*psclus->energy();
	  ps2_energy += (PFLayer::PS2 == psclus->layer())*psclus->energy();
	  ps_energy  += psclus->energy();
	  new_sc.addPreshowerCluster(psclus);
	}    
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

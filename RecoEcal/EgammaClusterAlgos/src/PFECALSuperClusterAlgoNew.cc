#include "RecoEcal/EgammaClusterAlgos/interface/PFECALSuperClusterAlgoNew.h"
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
  typedef PFECALSuperClusterAlgoNew::CalibratedClusterPtr CalibClusterPtr;
  typedef PFECALSuperClusterAlgoNew::CalibratedClusterPtrVector CalibClusterPtrVector;

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
    PFECALSuperClusterAlgoNew::clustering_type _type;
    double etawidthSuperCluster_, phiwidthSuperCluster_;
    IsClustered(const CalibClusterPtr s, 
		PFECALSuperClusterAlgoNew::clustering_type ct) : 
      the_seed(s), _type(ct) {}
    bool operator()(const CalibClusterPtr& x) { 
      double dphi = 
	std::abs(TVector2::Phi_mpi_pi(the_seed->phi() - x->phi()));  
      
      switch( _type ) {
      case PFECALSuperClusterAlgoNew::kBOX:
	return ( std::abs(the_seed->eta()-x->eta())<etawidthSuperCluster_ && 
		 dphi < phiwidthSuperCluster_   );
	break;
      case PFECALSuperClusterAlgoNew::kMustache:
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

PFECALSuperClusterAlgoNew::PFECALSuperClusterAlgoNew() {
}

void PFECALSuperClusterAlgoNew::
setPFClusterCalibration(const std::shared_ptr<PFEnergyCalibration>& calib) {
  _pfEnergyCalibration = calib;
}

void PFECALSuperClusterAlgoNew::
loadAndSortPFClusters(const PFClusterView& clusters,
		      const PFClusterView& psclusters) { 
  // reset the system for running
  superClustersEB_.reset(new reco::SuperClusterCollection);
  _clustersEB.clear();
  superClustersEE_.reset(new reco::SuperClusterCollection);  
  _clustersEE.clear();
  _clusteredEEclusters.clear();
  _psclusters.clear();
  _eetopsclusters.clear();
  
  auto clusterPtrs = clusters.ptrVector();
  _psclusters = psclusters.ptrVector();
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
      }
      break;
    default:
      break;
    }
  }
  // sort full cluster collections by their calibrated energy
  // this will put all the seeds first by construction
  GreaterByE greaterByE;
  std::sort(_clustersEB.begin(), _clustersEB.end(), greaterByE);
  std::sort(_clustersEE.begin(), _clustersEE.end(), greaterByE);  
}

void PFECALSuperClusterAlgoNew::run() {  
  // clusterize the EB
  buildAllSuperClusters(_clustersEB, threshPFClusterSeedBarrel_);
  // clusterize the EE
  buildAllSuperClusters(_clustersEE, threshPFClusterSeedEndcap_);
}

void PFECALSuperClusterAlgoNew::
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

void PFECALSuperClusterAlgoNew::
buildSuperCluster(CalibClusterPtr& seed,
		  CalibClusterPtrVector& clusters) {
  IsClustered IsClusteredWithSeed(seed,_clustype);
  bool isEE = false;   
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
    rawSCEnergy(0), corrSCEnergy(0);
  PFClusterPtrVector psclusters;
  for( auto clus : clustered ) {
    bare_ptrs.push_back(clus->the_ptr().get());
    if( isEE ) { 
      _clusteredEEclusters.push_back(clus);
      _eetopsclusters.insert(std::make_pair(clus->the_ptr(),
					    PFClusterPtrVector()));
    }
      
    posX += clus->energy_nocalib() * clus->the_ptr()->position().X();
    posY += clus->energy_nocalib() * clus->the_ptr()->position().Y();
    posZ += clus->energy_nocalib() * clus->the_ptr()->position().Z();    

    rawSCEnergy  += clus->energy_nocalib();
    corrSCEnergy += clus->energy();    
  }
  posX /= rawSCEnergy;
  posY /= rawSCEnergy;
  posZ /= rawSCEnergy;  
  // now build the supercluster
  reco::SuperCluster new_sc(corrSCEnergy,math::XYZPoint(posX,posY,posZ));
  new_sc.setSeed(clustered.front()->the_ptr());
  for( auto clus : clustered ) {
    new_sc.addCluster(clus->the_ptr());
    auto hits_and_fractions = clus->the_ptr()->hitsAndFractions();
    for( auto hit_and_fraction : hits_and_fractions ) {
      new_sc.addHitAndFraction(hit_and_fraction.first,hit_and_fraction.second);
    }    
  }  
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

void PFECALSuperClusterAlgoNew::addPSInfoToEESuperClusters() {
  // make the association map of ECAL clusters to preshower clusters    
  double dist = -1.0, min_dist = -1.0;
  // match PS clusters to clusters from the EE (many to one relation)
  for( auto psclus : _psclusters ) {
    if( psclus->energy() < threshPFClusterES_ ) continue;    
    switch( psclus->layer() ) { // just in case this isn't the ES...
    case PFLayer::PS1:
    case PFLayer::PS2:
      break;
    default:
      continue;
    }
    // for each ee cluster available, find the ps clusters that are
    // closest to it   
    PFClusterPtr eematch;
    min_dist = -1.0;
    for( auto eeclus : _clusteredEEclusters ) {
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
    _eetopsclusters[eematch].push_back(psclus);
  }  
  // update EE calibrated super cluster energies  
  SumPSEnergy sumps1(PFLayer::PS1), sumps2(PFLayer::PS2); 
  double clusterCorrEE, PS1_clus_sum, PS2_clus_sum, new_energy,
    ps1_energy, ps2_energy, ps_energy;
  for( auto sc : *superClustersEE_ ) {
    reco::CaloCluster_iterator clus = sc.clustersBegin();
    reco::CaloCluster_iterator clusend = sc.clustersEnd();
    new_energy = ps1_energy = ps2_energy = ps_energy = 0.0; // reset energies
    for(; clus != clusend; ++clus ) {
      PFClusterPtr casted_clus(*clus);
      auto psclusters = _eetopsclusters.find(casted_clus)->second;
      PS1_clus_sum = std::accumulate(psclusters.begin(),
				     psclusters.end(),
				     0.0,sumps1);
      PS2_clus_sum = std::accumulate(psclusters.begin(),
				     psclusters.end(),
				     0.0,sumps2);
      clusterCorrEE = 
	_pfEnergyCalibration->energyEm(*casted_clus,
				       PS1_clus_sum,
				       PS2_clus_sum,
				       applyCrackCorrections_);
      new_energy += clusterCorrEE;
      // compute PS energy and recalibrated energy for the EE supercluster
      auto cluspsassociation = _eetopsclusters.find(casted_clus);
      if( cluspsassociation != _eetopsclusters.end() ) {    
	// since EE rechits can share PS rechits, we have to make
	// sure the PS rechit hasn't already been added
	for( auto psclus : cluspsassociation->second ) {
	  auto found_pscluster = std::find(sc.preshowerClustersBegin(),
					   sc.preshowerClustersEnd(),
					   reco::CaloClusterPtr(psclus));
	  if( found_pscluster == sc.preshowerClustersEnd() ) {
	    ps1_energy += (PFLayer::PS1 == psclus->layer())*psclus->energy();
	    ps2_energy += (PFLayer::PS2 == psclus->layer())*psclus->energy();
	    ps_energy  += psclus->energy();
	    sc.addPreshowerCluster(psclus);
	  }    
	}
      }
    }    
    sc.setPreshowerEnergy(ps_energy); 
    sc.setPreshowerEnergyPlane1(ps1_energy);
    sc.setPreshowerEnergyPlane2(ps2_energy);
    //reset the SC's corrected energy
    sc.setEnergy(new_energy);
  }
}

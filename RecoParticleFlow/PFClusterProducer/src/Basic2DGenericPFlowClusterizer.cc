#include "Basic2DGenericPFlowClusterizer.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "Math/GenVector/VectorUtil.h"

#include "vdt/vdtMath.h"

#include <iterator>

#ifdef PFLOW_DEBUG
#define LOGVERB(x) edm::LogVerbatim(x)
#define LOGWARN(x) edm::LogWarning(x)
#define LOGERR(x) edm::LogError(x)
#define LOGDRESSED(x) edm::LogInfo(x)
#else
#define LOGVERB(x) LogTrace(x)
#define LOGWARN(x) edm::LogWarning(x)
#define LOGERR(x) edm::LogError(x)
#define LOGDRESSED(x) LogDebug(x)
#endif

Basic2DGenericPFlowClusterizer::
Basic2DGenericPFlowClusterizer(const edm::ParameterSet& conf) :
    PFClusterBuilderBase(conf),
    _maxIterations(conf.getParameter<unsigned>("maxIterations")),
    _stoppingTolerance(conf.getParameter<double>("stoppingTolerance")),
    _showerSigma2(std::pow(conf.getParameter<double>("showerSigma"),2.0)),
    _excludeOtherSeeds(conf.getParameter<bool>("excludeOtherSeeds")),
    _minFracTot(conf.getParameter<double>("minFracTot")),
    _layerMap({ {"PS2",(int)PFLayer::PS2},
	        {"PS1",(int)PFLayer::PS1},
	        {"ECAL_ENDCAP",(int)PFLayer::ECAL_ENDCAP},
	        {"ECAL_BARREL",(int)PFLayer::ECAL_BARREL},
	        {"NONE",(int)PFLayer::NONE},
	        {"HCAL_BARREL1",(int)PFLayer::HCAL_BARREL1},
	        {"HCAL_BARREL2_RING0",(int)PFLayer::HCAL_BARREL2},
		{"HCAL_BARREL2_RING1",100*(int)PFLayer::HCAL_BARREL2},
	        {"HCAL_ENDCAP",(int)PFLayer::HCAL_ENDCAP},
	        {"HF_EM",(int)PFLayer::HF_EM},
		{"HF_HAD",(int)PFLayer::HF_HAD} }) { 
  const std::vector<edm::ParameterSet>& thresholds =
    conf.getParameterSetVector("recHitEnergyNorms");
  for( const auto& pset : thresholds ) {
    const std::string& det = pset.getParameter<std::string>("detector");
    const double& rhE_norm = pset.getParameter<double>("recHitEnergyNorm");    
    auto entry = _layerMap.find(det);
    if( entry == _layerMap.end() ) {
      throw cms::Exception("InvalidDetectorLayer")
	<< "Detector layer : " << det << " is not in the list of recognized"
	<< " detector layers!";
    }
    _recHitEnergyNorms.emplace(_layerMap.find(det)->second,rhE_norm);
  }
  
  _allCellsPosCalc.reset(NULL);
  if( conf.exists("allCellsPositionCalc") ) {
    const edm::ParameterSet& acConf = 
      conf.getParameterSet("allCellsPositionCalc");
    const std::string& algoac = 
      acConf.getParameter<std::string>("algoName");
    PosCalc* accalc = 
      PFCPositionCalculatorFactory::get()->create(algoac, acConf);
    _allCellsPosCalc.reset(accalc);
  }
  // if necessary a third pos calc for convergence testing
  _convergencePosCalc.reset(NULL);
  if( conf.exists("positionCalcForConvergence") ) {
    const edm::ParameterSet& convConf = 
      conf.getParameterSet("positionCalcForConvergence");
    const std::string& algoconv = 
      convConf.getParameter<std::string>("algoName");
    PosCalc* convcalc = 
      PFCPositionCalculatorFactory::get()->create(algoconv, convConf);
    _convergencePosCalc.reset(convcalc);
  }
}

void Basic2DGenericPFlowClusterizer::
buildClusters(const reco::PFClusterCollection& input,
	      const std::vector<bool>& seedable,
	      reco::PFClusterCollection& output) {
  reco::PFClusterCollection clustersInTopo;
  for( const auto& topocluster : input ) {
    clustersInTopo.clear();
    seedPFClustersFromTopo(topocluster,seedable,clustersInTopo);
    const unsigned tolScal = 
      std::pow(std::max(1.0,clustersInTopo.size()-1.0),2.0);
    growPFClusters(topocluster,seedable,tolScal,0,tolScal,clustersInTopo);
    // step added by Josh Bendavid, removes low-fraction clusters
    // did not impact position resolution with fraction cut of 1e-7
    // decreases the size of each pf cluster considerably
    prunePFClusters(clustersInTopo);
    // recalculate the positions of the pruned clusters
    if( _convergencePosCalc ) { 
      // if defined, use the special position calculation for convergence tests
      _convergencePosCalc->calculateAndSetPositions(clustersInTopo);
    } else {
      if( clustersInTopo.size() == 1 && _allCellsPosCalc ) {
	_allCellsPosCalc->calculateAndSetPosition(clustersInTopo.back());
      } else {
	_positionCalc->calculateAndSetPositions(clustersInTopo);
      }   
    }
    for( auto& clusterout : clustersInTopo ) {
      output.insert(output.end(),std::move(clusterout));
    }
  }
}

void Basic2DGenericPFlowClusterizer::
seedPFClustersFromTopo(const reco::PFCluster& topo,
		       const std::vector<bool>& seedable,
		       reco::PFClusterCollection& initialPFClusters) const {
  const auto& recHitFractions = topo.recHitFractions();
  for( const auto& rhf : recHitFractions ) {
    if( !seedable[rhf.recHitRef().key()] ) continue;
    initialPFClusters.push_back(reco::PFCluster());
    reco::PFCluster& current = initialPFClusters.back();
    current.addRecHitFraction(rhf);
    current.setSeed(rhf.recHitRef()->detId());   
    if( _convergencePosCalc ) {
      _convergencePosCalc->calculateAndSetPosition(current);
    } else {
      _positionCalc->calculateAndSetPosition(current);
    }
  }
}

void Basic2DGenericPFlowClusterizer::
growPFClusters(const reco::PFCluster& topo,
	       const std::vector<bool>& seedable,
	       const unsigned toleranceScaling,
	       const unsigned iter,
	       double diff,
	       reco::PFClusterCollection& clusters) const {
  if( iter >= _maxIterations ) {
    LOGDRESSED("Basic2DGenericPFlowClusterizer:growAndStabilizePFClusters")
      <<"reached " << _maxIterations << " iterations, terminated position "
      << "fit with diff = " << diff;
  }      
  if( iter >= _maxIterations || 
      diff <= _stoppingTolerance*toleranceScaling) return;
  // reset the rechits in this cluster, keeping the previous position    
  std::vector<reco::PFCluster::REPPoint> clus_prev_pos;  
  for( auto& cluster : clusters) {
    const reco::PFCluster::REPPoint& repp = cluster.positionREP();
    clus_prev_pos.emplace_back(repp.rho(),repp.eta(),repp.phi());
    if( _convergencePosCalc ) {
      if( clusters.size() == 1 && _allCellsPosCalc ) {
	_allCellsPosCalc->calculateAndSetPosition(cluster);
      } else {
	_positionCalc->calculateAndSetPosition(cluster);
      }
    }
    cluster.resetHitsAndFractions();
  }
  // loop over topo cluster and grow current PFCluster hypothesis 
  std::vector<double> dist2, frac;
  double fractot = 0, fraction = 0;
  for( const reco::PFRecHitFraction& rhf : topo.recHitFractions() ) {
    const reco::PFRecHitRef& refhit = rhf.recHitRef();
    int cell_layer = (int)refhit->layer();
    if( cell_layer == PFLayer::HCAL_BARREL2 && 
	std::abs(refhit->positionREP().eta()) > 0.34 ) {
      cell_layer *= 100;
    }  
    const double recHitEnergyNorm = 
      _recHitEnergyNorms.find(cell_layer)->second; 
    const math::XYZPoint& topocellpos_xyz = refhit->position();
    dist2.clear(); frac.clear(); fractot = 0;
    // add rechits to clusters, calculating fraction based on distance
    for( auto& cluster : clusters ) {      
      const math::XYZPoint& clusterpos_xyz = cluster.position();
      fraction = 0.0;
      const math::XYZVector deltav = clusterpos_xyz - topocellpos_xyz;
      const double d2 = deltav.Mag2()/_showerSigma2;
      dist2.emplace_back( d2 );
      if( d2 > 100 ) {
	LOGDRESSED("Basic2DGenericPFlowClusterizer:growAndStabilizePFClusters")
	  << "Warning! :: pfcluster-topocell distance is too large! d= "
	  << d2;
      }
      // fraction assignment logic
      if( refhit->detId() == cluster.seed() && _excludeOtherSeeds ) {
	fraction = 1.0;	
      } else if ( seedable[refhit.key()] && _excludeOtherSeeds ) {
	fraction = 0.0;
      } else {
	fraction = cluster.energy()/recHitEnergyNorm * vdt::fast_expf( -0.5*d2 );
      }      
      fractot += fraction;
      frac.emplace_back(fraction);
    }
    for( unsigned i = 0; i < clusters.size(); ++i ) {      
      if( fractot > _minFracTot || 
	  ( refhit->detId() == clusters[i].seed() && fractot > 0.0 ) ) {
	frac[i]/=fractot;
      } else {
	continue;
      }
      // if the fraction has been set to 0, the cell 
      // is now added to the cluster - careful ! (PJ, 19/07/08)
      // BUT KEEP ONLY CLOSE CELLS OTHERWISE MEMORY JUST EXPLOSES
      // (PJ, 15/09/08 <- similar to what existed before the 
      // previous bug fix, but keeps the close seeds inside, 
      // even if their fraction was set to zero.)
      // Also add a protection to keep the seed in the cluster 
      // when the latter gets far from the former. These cases
      // (about 1% of the clusters) need to be studied, as 
      // they create fake photons, in general.
      // (PJ, 16/09/08) 
      if( dist2[i] < 100.0 || frac[i] > 0.9999 ) {	
	clusters[i].addRecHitFraction(reco::PFRecHitFraction(refhit,frac[i]));
      }
    }
  }
  // recalculate positions and calculate convergence parameter
  double diff2 = 0.0;  
  for( unsigned i = 0; i < clusters.size(); ++i ) {
    if( _convergencePosCalc ) {
      _convergencePosCalc->calculateAndSetPosition(clusters[i]);
    } else {
      if( clusters.size() == 1 && _allCellsPosCalc ) {
	_allCellsPosCalc->calculateAndSetPosition(clusters[i]);
      } else {
	_positionCalc->calculateAndSetPosition(clusters[i]);
      }
    }
    const double delta2 = 
      reco::deltaR2(clusters[i].positionREP(),clus_prev_pos[i]);    
    if( delta2 > diff2 ) diff2 = delta2;
  }
  diff = std::sqrt(diff2);
  dist2.clear(); frac.clear(); clus_prev_pos.clear();// avoid badness
  growPFClusters(topo,seedable,toleranceScaling,iter+1,diff,clusters);
}

void Basic2DGenericPFlowClusterizer::
prunePFClusters(reco::PFClusterCollection& clusters) const {
  for( auto& cluster : clusters ) {
    cluster.pruneUsing( [&](const reco::PFRecHitFraction& rhf)
			{return rhf.fraction() > _minFractionToKeep;} 
			);    
  }
}



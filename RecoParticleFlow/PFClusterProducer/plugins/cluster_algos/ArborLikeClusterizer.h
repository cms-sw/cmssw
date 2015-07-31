#ifndef __ArborLikeClusterizer_H__
#define __ArborLikeClusterizer_H__

#include "RecoParticleFlow/PFClusterProducer/interface/InitialClusteringStepBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"
#include "RecoParticleFlow/PFClusterProducer/interface/Arbor.hh"

#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"

class ArborLikeClusterizer : public InitialClusteringStepBase {  
 public:
  typedef ArborLikeClusterizer B2DGT;
  typedef PFCPositionCalculatorBase PosCalc;
  typedef std::unordered_map<unsigned,bool> seed_usage_map;
  typedef std::unordered_map<unsigned,std::unordered_map<unsigned,double> > seed_fractions_map;
  
  enum seed_type{ NotSeed = 0, PrimarySeed=1, SecondarySeed=2 };
  enum navi_dir{ Bidirectional = 0, OnlyForward = 1, OnlyBackward = 2};
  
  
  ArborLikeClusterizer(const edm::ParameterSet& conf,
		       edm::ConsumesCollector& sumes);
  virtual ~ArborLikeClusterizer() {}
  ArborLikeClusterizer(const B2DGT&) = delete;
  B2DGT& operator=(const B2DGT&) = delete;

  void buildClusters(const edm::Handle<reco::PFRecHitCollection>&,
		     const std::vector<bool>&,
		     const std::vector<bool>&, 
		     reco::PFClusterCollection&);
  
 private:  
  const bool _useCornerCells;
  const double _showerSigma2,_stoppingTolerance,_minFracTot;
  const unsigned _maxIterations;

  std::unique_ptr<PFCPositionCalculatorBase> _positionCalc;
  std::unique_ptr<PFCPositionCalculatorBase> _allCellsPosCalc;
  
  void linkSeeds(const reco::PFRecHitCollection&,
		 const std::vector<bool>&,
		 const std::vector<std::pair<unsigned,double> >&,
		 std::unordered_multimap<unsigned,unsigned>&,
		 std::vector<seed_type>&,
		 std::vector<std::vector<unsigned> >&) const;
  
  void findSeedNeighbours(const reco::PFRecHitCollection&,
			  const std::vector<bool>&,
			  const unsigned,
			  const unsigned,
			  std::vector<bool>&,
			  std::unordered_multimap<unsigned,unsigned>&,
			  std::vector<unsigned>&,
			  navi_dir direction = Bidirectional) const;

  

  void buildTopoCluster(const edm::Handle<reco::PFRecHitCollection>&,
			const std::vector<bool>&, // masked rechits
			const reco::PFRecHitRef&, //present rechit
			std::vector<bool>&, // hit usage state
			reco::PFCluster&); // the topocluster

  void buildInitialWeightsList(const reco::PFRecHitCollection& rechits,
			       const std::vector<bool>& seedable,
			       const std::vector<seed_type>& seedtypes,
			       const std::vector<std::vector<unsigned> >& linked_seeds,
			       const unsigned  seed_idx,
			       seed_usage_map& has_weight_data,
			       seed_fractions_map& resolved_seeds) const;
  
  void getLinkedTopoClusters(const std::unordered_multimap<unsigned,unsigned>&,
			     const std::unordered_multimap<unsigned,unsigned>&,
			     const reco::PFClusterCollection&,
			     const unsigned,
			     std::vector<bool>&,
			     std::vector<unsigned>&) const;

  void growPFClusters(const std::vector<std::vector<unsigned> >&,
		      const std::vector<std::vector<unsigned> >&,
		      const std::unordered_multimap<unsigned,unsigned>&,
		      const std::vector<bool>&,		      
		      const reco::PFClusterCollection& topoclusters,
		      const unsigned,
		      double,
		      reco::PFClusterCollection&) const;
  

  
  
};

DEFINE_EDM_PLUGIN(InitialClusteringStepFactory,
		  ArborLikeClusterizer,
		  "ArborLikeClusterizer");

#endif

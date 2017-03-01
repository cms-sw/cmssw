#ifndef RecoParticleFlow_PFProducer_PFBlockAlgo_h
#define RecoParticleFlow_PFProducer_PFBlockAlgo_h 

#include <set>
#include <vector>
#include <iostream>

// #include "FWCore/Framework/interface/Handle.h"
#include "DataFormats/Common/interface/Handle.h"
// #include "FWCore/Framework/interface/OrphanHandle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"


#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedTrackerVertex.h" // gouzevitch
#include "DataFormats/ParticleFlowReco/interface/PFConversionFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFConversion.h"
#include "DataFormats/ParticleFlowReco/interface/PFV0Fwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFV0.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBrem.h"
#include "DataFormats/ParticleFlowReco/interface/PFTrajectoryPoint.h"

#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockFwd.h"

// Glowinski & Gouzevitch
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"             
#include "RecoParticleFlow/PFProducer/interface/KDTreeLinkerBase.h" 
// !Glowinski & Gouzevitch

// #include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "RecoParticleFlow/PFProducer/interface/PFMuonAlgo.h"
#include "RecoParticleFlow/PFProducer/interface/PhotonSelectorAlgo.h"
#include "RecoParticleFlow/PFProducer/interface/PFBlockElementSCEqual.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFResolutionMap.h"

#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementSuperCluster.h"

#include "RecoParticleFlow/PFClusterTools/interface/ClusterClusterMapping.h"

#include "RecoParticleFlow/PFProducer/interface/PFBlockLink.h"

#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
#include "RecoParticleFlow/PFProducer/interface/BlockElementImporterBase.h"

#include <map>
#include <unordered_map>
#include <unordered_set>

namespace std {
  template<>
    struct hash<std::pair<unsigned int,unsigned int> > {
    typedef std::pair<unsigned int,unsigned int> arg_type;
    typedef unsigned int value_type;
    value_type operator()(const arg_type& arg) const {
      return arg.first ^ (arg.second << 1);
    }
  };
  template<>
    struct equal_to<std::pair<unsigned int,unsigned int> > {
    typedef std::pair<unsigned int,unsigned int> arg_type;    
    bool operator()(const arg_type& arg1, const arg_type& arg2) const {
      return ( (arg1.first == arg2.first) & (arg1.second == arg2.second) );
    }
  };
}

/// \brief Particle Flow Algorithm
/*!
  \author Colin Bernet (rewrite/refactor by L. Gray)
  \date January 2006 (April 2014) 
*/

class PFBlockAlgo {

 public:
  // the element list should **always** be a list of (smart) pointers
  typedef std::vector<std::unique_ptr<reco::PFBlockElement> > ElementList;
  typedef std::unique_ptr<BlockElementImporterBase> ImporterPtr;
  typedef std::unique_ptr<BlockElementLinkerBase> LinkTestPtr;  
  typedef std::unique_ptr<KDTreeLinkerBase> KDTreePtr;
  /// define these in *Fwd files in DataFormats/ParticleFlowReco?
  typedef ElementList::iterator IE;
  typedef ElementList::const_iterator IEC;  
  typedef reco::PFBlockCollection::const_iterator IBC;
  //for skipping ranges
  typedef std::array<std::pair<unsigned int,unsigned int>,reco::PFBlockElement::kNBETypes> ElementRanges;
  
  PFBlockAlgo();

  ~PFBlockAlgo();
    
  void setLinkers(const std::vector<edm::ParameterSet>&);

  void setImporters(const std::vector<edm::ParameterSet>&,
		    edm::ConsumesCollector&);
  
  // update event setup info of all linkers
  void updateEventSetup(const edm::EventSetup&);

  // run all of the importers and build KDtrees
  void buildElements(const edm::Event&);
  
  /// build blocks
  void findBlocks();

  /// sets debug printout flag
  void setDebug( bool debug ) {debug_ = debug;}
  
  /// \return collection of blocks
  /*   const  reco::PFBlockCollection& blocks() const {return *blocks_;} */
  const std::unique_ptr< reco::PFBlockCollection >& blocks() const 
    {return blocks_;}
  
  //uniquen unique_ptr to collection of blocks
  std::unique_ptr< reco::PFBlockCollection > transferBlocks() {return std::move(blocks_);}

  
  
 private:
  
  /// compute missing links in the blocks 
  /// (the recursive procedure does not build all links)  
  void packLinks(reco::PFBlock& block, 
		 const std::unordered_map<std::pair<unsigned int,unsigned int>,PFBlockLink>& links) const; 
  
  /// Avoid to check links when not useful
  inline bool linkPrefilter(const reco::PFBlockElement* last, 
			    const reco::PFBlockElement* next) const;

  /// check whether 2 elements are linked. Returns distance and linktype
  inline void link( const reco::PFBlockElement* el1, 
		    const reco::PFBlockElement* el2, 
		    PFBlockLink::Type& linktype, 
		    reco::PFBlock::LinkTest& linktest,
		    double& dist) const;
  
  std::unique_ptr< reco::PFBlockCollection >    blocks_;
  
  // the test elements will be transferred to the blocks
  ElementList       elements_; 
  std::vector<ElementList::value_type::pointer> bare_elements_;
  ElementRanges     ranges_;
  
  /// if true, debug printouts activated
  bool   debug_;
  
  friend std::ostream& operator<<(std::ostream&, const PFBlockAlgo&);
  bool useHO_;

  std::vector<ImporterPtr> importers_;

  const std::unordered_map<std::string,reco::PFBlockElement::Type> 
    elementTypes_;
  std::vector<LinkTestPtr> linkTests_;
  unsigned int linkTestSquare_[reco::PFBlockElement::kNBETypes][reco::PFBlockElement::kNBETypes];
  
  std::vector<KDTreePtr> kdtrees_;
};

#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementBrem.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"

#endif



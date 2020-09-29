#ifndef RecoParticleFlow_PFProducer_PFBlockAlgo_h
#define RecoParticleFlow_PFProducer_PFBlockAlgo_h

#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockFwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "RecoParticleFlow/PFProducer/interface/BlockElementImporterBase.h"
#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
#include "RecoParticleFlow/PFProducer/interface/KDTreeLinkerBase.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "RecoParticleFlow/PFProducer/interface/PFTables.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace std {
  template <>
  struct hash<std::pair<unsigned int, unsigned int>> {
    typedef std::pair<unsigned int, unsigned int> arg_type;
    typedef unsigned int value_type;
    value_type operator()(const arg_type& arg) const { return arg.first ^ (arg.second << 1); }
  };
}  // namespace std

/// \brief Particle Flow Algorithm
/*!
  \author Colin Bernet (rewrite/refactor by L. Gray)
  \date January 2006 (April 2014) 
*/

class PFBlockAlgo {
public:
  // the element list should **always** be a list of (smart) pointers
  //for skipping ranges
  typedef std::array<std::pair<unsigned int, unsigned int>, reco::PFBlockElement::kNBETypes> ElementRanges;

  PFBlockAlgo();

  ~PFBlockAlgo();

  void setLinkers(const std::vector<edm::ParameterSet>&);

  void setImporters(const std::vector<edm::ParameterSet>&, edm::ConsumesCollector&);

  // update event setup info of all linkers
  void updateEventSetup(const edm::EventSetup&);

  // run all of the importers and build KDtrees
  const PFTables buildElements(const edm::Event&);

  /// build blocks
  reco::PFBlockCollection findBlocks(const PFTables& tables);

  /// sets debug printout flag
  void setDebug(bool debug) { debug_ = debug; }

private:
  /// compute missing links in the blocks
  /// (the recursive procedure does not build all links)
  //block_element_indices - element indices (in the full element list) assigned to this block
  //links - link data that is already computed in this block
  const reco::PFBlock packLinks(const PFTables& tables,
                                const std::vector<size_t>& block_element_indices,
                                const std::unordered_map<std::pair<unsigned int, unsigned int>, double>& links,
                                const ElementListConst& elements_,
                                const reco::PFMultiLinksIndex& multilinks) const;

  /// check whether 2 elements are linked. Returns distance.
  //iel1 - index of the first element in the full element list
  //iel2 - index of the second element in the full element list
  //dist - computed distance between elements (output)
  //elements_ - needed only for the legacy non-SOA evaluation of distance
  //tables - read-only SOA data of the PFBlockElements
  //multilinks - read-only link data between elements computed by KDTreeLinkers
  inline void link(size_t iel1,
                   size_t iel2,
                   double& dist,
                   const ElementListConst& elements_,
                   const PFTables& tables,
                   const reco::PFMultiLinksIndex& multilinks) const;

  // the test elements will be transferred to the blocks
  ElementList elements_;
  ElementRanges ranges_;

  /// if true, debug printouts activated
  bool debug_;

  friend std::ostream& operator<<(std::ostream&, const PFBlockAlgo&);
  bool useHO_;

  std::vector<std::unique_ptr<BlockElementImporterBase>> importers_;

  const std::unordered_map<std::string, reco::PFBlockElement::Type> elementTypes_;
  std::vector<std::unique_ptr<BlockElementLinkerBase>> linkTests_;
  unsigned int linkTestSquare_[reco::PFBlockElement::kNBETypes][reco::PFBlockElement::kNBETypes];

  std::vector<std::unique_ptr<KDTreeLinkerBase>> kdtrees_;

  // rechit with fraction below this value will be ignored in KDTreeLinker
  static constexpr float cutOffFrac_ = 1E-4;
};

#endif

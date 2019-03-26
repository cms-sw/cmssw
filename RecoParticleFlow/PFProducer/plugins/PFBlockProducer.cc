#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementBrem.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementSuperCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBrem.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFConversion.h"
#include "DataFormats/ParticleFlowReco/interface/PFConversionFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedTrackerVertex.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertex.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFTrajectoryPoint.h"
#include "DataFormats/ParticleFlowReco/interface/PFV0.h"
#include "DataFormats/ParticleFlowReco/interface/PFV0Fwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "RecoParticleFlow/PFClusterTools/interface/ClusterClusterMapping.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyResolution.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFResolutionMap.h"
#include "RecoParticleFlow/PFProducer/interface/BlockElementImporterBase.h"
#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
#include "RecoParticleFlow/PFProducer/interface/KDTreeLinkerBase.h"
#include "RecoParticleFlow/PFProducer/interface/PFBlockElementSCEqual.h"
#include "RecoParticleFlow/PFProducer/interface/PFBlockLink.h"
#include "RecoParticleFlow/PFProducer/interface/PFMuonAlgo.h"
#include "RecoParticleFlow/PFProducer/interface/PhotonSelectorAlgo.h"
#include "RecoParticleFlow/PFProducer/interface/Utils.h"

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "TMath.h"

namespace std {
  template <> struct hash<std::pair<unsigned int, unsigned int> > {
    typedef std::pair<unsigned int, unsigned int> arg_type;
    typedef unsigned int value_type;
    value_type operator()(const arg_type& arg) const { return arg.first ^ (arg.second << 1); }
  };
  template <> struct equal_to<std::pair<unsigned int, unsigned int> > {
    typedef std::pair<unsigned int, unsigned int> arg_type;
    bool operator()(const arg_type& arg1, const arg_type& arg2) const {
      return ((arg1.first == arg2.first) & (arg1.second == arg2.second));
    }
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
  typedef std::vector<std::unique_ptr<reco::PFBlockElement> > ElementList;
  typedef std::unique_ptr<BlockElementImporterBase> ImporterPtr;
  typedef std::unique_ptr<BlockElementLinkerBase> LinkTestPtr;
  typedef std::unique_ptr<KDTreeLinkerBase> KDTreePtr;
  /// define these in *Fwd files in DataFormats/ParticleFlowReco?
  typedef ElementList::iterator IE;
  typedef ElementList::const_iterator IEC;
  typedef reco::PFBlockCollection::const_iterator IBC;
  //for skipping ranges
  typedef std::array<std::pair<unsigned int, unsigned int>, reco::PFBlockElement::kNBETypes> ElementRanges;

  PFBlockAlgo();

  ~PFBlockAlgo();

  void setLinkers(const std::vector<edm::ParameterSet>&);

  void setImporters(const std::vector<edm::ParameterSet>&, edm::ConsumesCollector&);

  // update event setup info of all linkers
  void updateEventSetup(const edm::EventSetup&);

  // run all of the importers and build KDtrees
  void buildElements(const edm::Event&);

  /// build blocks
  reco::PFBlockCollection findBlocks();

  /// sets debug printout flag
  void setDebug(bool debug) { debug_ = debug; }

private:
  /// compute missing links in the blocks
  /// (the recursive procedure does not build all links)
  void packLinks(reco::PFBlock& block,
                 const std::unordered_map<std::pair<unsigned int, unsigned int>, PFBlockLink>& links) const;

  /// Avoid to check links when not useful
  inline bool linkPrefilter(const reco::PFBlockElement* last, const reco::PFBlockElement* next) const;

  /// check whether 2 elements are linked. Returns distance and linktype
  inline void link(const reco::PFBlockElement* el1,
                   const reco::PFBlockElement* el2,
                   PFBlockLink::Type& linktype,
                   reco::PFBlock::LinkTest& linktest,
                   double& dist) const;

  // the test elements will be transferred to the blocks
  ElementList elements_;
  ElementRanges ranges_;

  /// if true, debug printouts activated
  bool debug_;

  friend std::ostream& operator<<(std::ostream&, const PFBlockAlgo&);
  bool useHO_;

  std::vector<ImporterPtr> importers_;

  const std::unordered_map<std::string, reco::PFBlockElement::Type> elementTypes_;
  std::vector<LinkTestPtr> linkTests_;
  unsigned int linkTestSquare_[reco::PFBlockElement::kNBETypes][reco::PFBlockElement::kNBETypes];

  std::vector<KDTreePtr> kdtrees_;
};

using namespace std;
using namespace reco;

#define INIT_ENTRY(name) \
  { #name, name }

namespace {
  class QuickUnion {
    std::vector<unsigned> id_;
    std::vector<unsigned> size_;
    int count_;

  public:
    QuickUnion(const unsigned NBranches) {
      count_ = NBranches;
      id_.resize(NBranches);
      size_.resize(NBranches);
      for (unsigned i = 0; i < NBranches; ++i) {
        id_[i] = i;
        size_[i] = 1;
      }
    }

    int count() const { return count_; }

    unsigned find(unsigned p) {
      while (p != id_[p]) {
        id_[p] = id_[id_[p]];
        p = id_[p];
      }
      return p;
    }

    bool connected(unsigned p, unsigned q) { return find(p) == find(q); }

    void unite(unsigned p, unsigned q) {
      unsigned rootP = find(p);
      unsigned rootQ = find(q);
      id_[p] = q;

      if (size_[rootP] < size_[rootQ]) {
        id_[rootP] = rootQ;
        size_[rootQ] += size_[rootP];
      } else {
        id_[rootQ] = rootP;
        size_[rootP] += size_[rootQ];
      }
      --count_;
    }
  };
}  // namespace

//for debug only
//#define PFLOW_DEBUG

PFBlockAlgo::PFBlockAlgo()
    : debug_(false),
      elementTypes_({INIT_ENTRY(PFBlockElement::TRACK),
                     INIT_ENTRY(PFBlockElement::PS1),
                     INIT_ENTRY(PFBlockElement::PS2),
                     INIT_ENTRY(PFBlockElement::ECAL),
                     INIT_ENTRY(PFBlockElement::HCAL),
                     INIT_ENTRY(PFBlockElement::GSF),
                     INIT_ENTRY(PFBlockElement::BREM),
                     INIT_ENTRY(PFBlockElement::HFEM),
                     INIT_ENTRY(PFBlockElement::HFHAD),
                     INIT_ENTRY(PFBlockElement::SC),
                     INIT_ENTRY(PFBlockElement::HO),
                     INIT_ENTRY(PFBlockElement::HGCAL)}) {}

void PFBlockAlgo::setLinkers(const std::vector<edm::ParameterSet>& confs) {
  constexpr unsigned rowsize = reco::PFBlockElement::kNBETypes;
  for (unsigned i = 0; i < rowsize; ++i) {
    for (unsigned j = 0; j < rowsize; ++j) {
      linkTestSquare_[i][j] = 0;
    }
  }
  linkTests_.resize(rowsize * rowsize);
  const std::string prefix("PFBlockElement::");
  const std::string pfx_kdtree("KDTree");
  for (const auto& conf : confs) {
    const std::string& linkerName = conf.getParameter<std::string>("linkerName");
    const std::string& linkTypeStr = conf.getParameter<std::string>("linkType");
    size_t split = linkTypeStr.find(':');
    if (split == std::string::npos) {
      throw cms::Exception("MalformedLinkType") << "\"" << linkTypeStr << "\" is not a valid link type definition."
                                                << " This string should have the form \"linkFrom:linkTo\"";
    }
    std::string link1(prefix + linkTypeStr.substr(0, split));
    std::string link2(prefix + linkTypeStr.substr(split + 1, std::string::npos));
    if (!(elementTypes_.count(link1) && elementTypes_.count(link2))) {
      throw cms::Exception("InvalidBlockElementType")
          << "One of \"" << link1 << "\" or \"" << link2 << "\" are invalid block element types!";
    }
    const PFBlockElement::Type type1 = elementTypes_.at(link1);
    const PFBlockElement::Type type2 = elementTypes_.at(link2);
    const unsigned index = rowsize * std::max(type1, type2) + std::min(type1, type2);
    linkTests_[index] = LinkTestPtr{BlockElementLinkerFactory::get()->create(linkerName, conf)};
    linkTestSquare_[type1][type2] = index;
    linkTestSquare_[type2][type1] = index;
    // setup KDtree if requested
    const bool useKDTree = conf.getParameter<bool>("useKDTree");
    if (useKDTree) {
      kdtrees_.emplace_back(KDTreeLinkerFactory::get()->create(pfx_kdtree + linkerName));
      kdtrees_.back()->setTargetType(std::min(type1, type2));
      kdtrees_.back()->setFieldType(std::max(type1, type2));
    }
  }
}

void PFBlockAlgo::setImporters(const std::vector<edm::ParameterSet>& confs, edm::ConsumesCollector& sumes) {
  importers_.reserve(confs.size());
  for (const auto& conf : confs) {
    const std::string& importerName = conf.getParameter<std::string>("importerName");
    importers_.emplace_back(BlockElementImporterFactory::get()->create(importerName, conf, sumes));
  }
}

PFBlockAlgo::~PFBlockAlgo() {
#ifdef PFLOW_DEBUG
  if (debug_)
    cout << "~PFBlockAlgo - number of remaining elements: " << elements_.size() << endl;
#endif
}

reco::PFBlockCollection PFBlockAlgo::findBlocks() {
  // Glowinski & Gouzevitch
  for (const auto& kdtree : kdtrees_) {
    kdtree->process();
  }
  // !Glowinski & Gouzevitch
  reco::PFBlockCollection blocks;
  // the blocks have not been passed to the event, and need to be cleared
  blocks.reserve(elements_.size());

  QuickUnion qu(elements_.size());
  const auto elem_size = elements_.size();
  for (unsigned i = 0; i < elem_size; ++i) {
    for (unsigned j = 0; j < elem_size; ++j) {
      if (qu.connected(i, j) || j == i)
        continue;
      if (!linkTests_[linkTestSquare_[elements_[i]->type()][elements_[j]->type()]]) {
        j = ranges_[elements_[j]->type()].second;
        continue;
      }
      auto p1(elements_[i].get()), p2(elements_[j].get());
      const PFBlockElement::Type type1 = p1->type();
      const PFBlockElement::Type type2 = p2->type();
      const unsigned index = linkTestSquare_[type1][type2];
      if (linkTests_[index]->linkPrefilter(p1, p2)) {
        const double dist = linkTests_[index]->testLink(p1, p2);
        // compute linking info if it is possible
        if (dist > -0.5) {
          qu.unite(i, j);
        }
      }
    }
  }

  std::unordered_multimap<unsigned, unsigned> blocksmap(elements_.size());
  std::vector<unsigned> keys;
  keys.reserve(elements_.size());
  for (unsigned i = 0; i < elements_.size(); ++i) {
    unsigned key = i;
    while (key != qu.find(key))
      key = qu.find(key);  // make sure we always find the root node...
    auto pos = std::lower_bound(keys.begin(), keys.end(), key);
    if (pos == keys.end() || *pos != key) {
      keys.insert(pos, key);
    }
    blocksmap.emplace(key, i);
  }

  PFBlockLink::Type linktype = PFBlockLink::NONE;
  PFBlock::LinkTest linktest = PFBlock::LINKTEST_RECHIT;
  for (auto key : keys) {
    blocks.push_back(reco::PFBlock());
    auto range = blocksmap.equal_range(key);
    auto& the_block = blocks.back();
    ElementList::value_type::pointer p1(elements_[range.first->second].get());
    the_block.addElement(p1);
    const unsigned block_size = blocksmap.count(key) + 1;
    //reserve up to 1M or 8MB; pay rehash cost for more
    std::unordered_map<std::pair<unsigned int, unsigned int>, PFBlockLink> links(
        min(1000000u, block_size * block_size));
    auto itr = range.first;
    ++itr;
    for (; itr != range.second; ++itr) {
      ElementList::value_type::pointer p2(elements_[itr->second].get());
      const PFBlockElement::Type type1 = p1->type();
      const PFBlockElement::Type type2 = p2->type();
      the_block.addElement(p2);
      linktest = PFBlock::LINKTEST_RECHIT;  //rechit by default
      linktype = static_cast<PFBlockLink::Type>(1 << (type1 - 1) | 1 << (type2 - 1));
      const unsigned index = linkTestSquare_[type1][type2];
      if (nullptr != linkTests_[index]) {
        const double dist = linkTests_[index]->testLink(p1, p2);
        links.emplace(std::make_pair(p1->index(), p2->index()),
                      PFBlockLink(linktype, linktest, dist, p1->index(), p2->index()));
      }
    }
    packLinks(the_block, links);
  }

  elements_.clear();

  return blocks;
}

void PFBlockAlgo::packLinks(reco::PFBlock& block,
                            const std::unordered_map<std::pair<unsigned int, unsigned int>, PFBlockLink>& links) const {
  constexpr unsigned rowsize = reco::PFBlockElement::kNBETypes;

  const edm::OwnVector<reco::PFBlockElement>& els = block.elements();

  block.bookLinkData();
  unsigned elsize = els.size();
  //First Loop: update all link data
  for (unsigned i1 = 0; i1 < elsize; ++i1) {
    for (unsigned i2 = 0; i2 < i1; ++i2) {
      // no reflexive link
      //if( i1==i2 ) continue;

      double dist = -1;

      bool linked = false;
      PFBlock::LinkTest linktest = PFBlock::LINKTEST_RECHIT;

      // are these elements already linked ?
      // this can be optimized
      const auto link_itr = links.find(std::make_pair(i2, i1));
      if (link_itr != links.end()) {
        dist = link_itr->second.dist();
        linktest = link_itr->second.test();
        linked = true;
      }

      if (!linked) {
        const PFBlockElement::Type type1 = els[i1].type();
        const PFBlockElement::Type type2 = els[i2].type();
        const auto minmax = std::minmax(type1, type2);
        const unsigned index = rowsize * minmax.second + minmax.first;
        PFBlockLink::Type linktype = PFBlockLink::NONE;
        bool bTestLink =
            (nullptr == linkTests_[index] ? false : linkTests_[index]->linkPrefilter(&(els[i1]), &(els[i2])));
        if (bTestLink)
          link(&els[i1], &els[i2], linktype, linktest, dist);
      }

      //loading link data according to link test used: RECHIT
      //block.setLink( i1, i2, chi2, block.linkData() );
#ifdef PFLOW_DEBUG
      if (debug_)
        cout << "Setting link between elements " << i1 << " and " << i2 << " of dist =" << dist
             << " computed from link test " << linktest << endl;
#endif
      block.setLink(i1, i2, dist, block.linkData(), linktest);
    }
  }
}

// see plugins/linkers for the functions that calculate distances
// for each available link type
inline bool PFBlockAlgo::linkPrefilter(const reco::PFBlockElement* last, const reco::PFBlockElement* next) const {
  constexpr unsigned rowsize = reco::PFBlockElement::kNBETypes;
  const PFBlockElement::Type type1 = (last)->type();
  const PFBlockElement::Type type2 = (next)->type();
  const unsigned index = rowsize * std::max(type1, type2) + std::min(type1, type2);
  bool result = linkTests_[index]->linkPrefilter(last, next);
  return result;
}

inline void PFBlockAlgo::link(const reco::PFBlockElement* el1,
                              const reco::PFBlockElement* el2,
                              PFBlockLink::Type& linktype,
                              reco::PFBlock::LinkTest& linktest,
                              double& dist) const {
  constexpr unsigned rowsize = reco::PFBlockElement::kNBETypes;
  dist = -1.0;
  linktest = PFBlock::LINKTEST_RECHIT;  //rechit by default
  const PFBlockElement::Type type1 = el1->type();
  const PFBlockElement::Type type2 = el2->type();
  linktype = static_cast<PFBlockLink::Type>(1 << (type1 - 1) | 1 << (type2 - 1));
  const unsigned index = rowsize * std::max(type1, type2) + std::min(type1, type2);
  if (debug_) {
    std::cout << " PFBlockAlgo links type1 " << type1 << " type2 " << type2 << std::endl;
  }

  // index is always checked in the preFilter above, no need to check here
  dist = linkTests_[index]->testLink(el1, el2);
}

void PFBlockAlgo::updateEventSetup(const edm::EventSetup& es) {
  for (auto& importer : importers_) {
    importer->updateEventSetup(es);
  }
}

// see plugins/importers and plugins/kdtrees
// for the definitions of available block element importers
// and kdtree preprocessors
void PFBlockAlgo::buildElements(const edm::Event& evt) {
  // import block elements as defined in python configuration
  ranges_.fill(std::make_pair(0, 0));
  elements_.clear();
  for (const auto& importer : importers_) {
    importer->importToBlock(evt, elements_);
  }

  std::sort(elements_.begin(), elements_.end(), [](const auto& a, const auto& b) { return a->type() < b->type(); });

  // list is now partitioned, so mark the boundaries so we can efficiently skip chunks
  unsigned current_type = (!elements_.empty() ? elements_[0]->type() : 0);
  unsigned last_type = (!elements_.empty() ? elements_.back()->type() : 0);
  ranges_[current_type].first = 0;
  ranges_[last_type].second = elements_.size() - 1;
  for (size_t i = 0; i < elements_.size(); ++i) {
    const auto the_type = elements_[i]->type();
    if (the_type != current_type) {
      ranges_[the_type].first = i;
      ranges_[current_type].second = i - 1;
      current_type = the_type;
    }
  }
  // -------------- Loop over block elements ---------------------

  // Here we provide to all KDTree linkers the collections to link.
  // Glowinski & Gouzevitch

  for (ElementList::iterator it = elements_.begin(); it != elements_.end(); ++it) {
    for (const auto& kdtree : kdtrees_) {
      if ((*it)->type() == kdtree->targetType()) {
        kdtree->insertTargetElt(it->get());
      }
      if ((*it)->type() == kdtree->fieldType()) {
        kdtree->insertFieldClusterElt(it->get());
      }
    }
  }
  //std::cout << "(new) imported: " << elements_.size() << " elements!" << std::endl;
}

std::ostream& operator<<(std::ostream& out, const PFBlockAlgo& a) {
  if (!out)
    return out;

  out << "====== Particle Flow Block Algorithm ======= ";
  out << endl;
  out << "number of unassociated elements : " << a.elements_.size() << endl;
  out << endl;

  for (PFBlockAlgo::IEC ie = a.elements_.begin(); ie != a.elements_.end(); ++ie) {
    out << "\t" << **ie << endl;
  }

  return out;
}

/**\class PFBlockProducer
\brief Producer for particle flow blocks

This producer makes use of PFBlockAlgo, the particle flow block algorithm.
Particle flow itself consists in reconstructing particles from the particle
flow blocks This is done at a later stage, see PFProducer and PFAlgo.

\author Colin Bernet
\date   April 2007
*/

class FSimEvent;

class PFBlockProducer : public edm::stream::EDProducer<> {
public:
  explicit PFBlockProducer(const edm::ParameterSet&);

  ~PFBlockProducer() override;

  void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  /// verbose ?
  const bool verbose_;
  const edm::EDPutTokenT<reco::PFBlockCollection> putToken_;

  /// Particle flow block algorithm
  PFBlockAlgo pfBlockAlgo_;
};

DEFINE_FWK_MODULE(PFBlockProducer);

using namespace std;
using namespace edm;

PFBlockProducer::PFBlockProducer(const edm::ParameterSet& iConfig)
    : verbose_{iConfig.getUntrackedParameter<bool>("verbose", false)}, putToken_{produces<reco::PFBlockCollection>()} {
  bool debug_ = iConfig.getUntrackedParameter<bool>("debug", false);
  pfBlockAlgo_.setDebug(debug_);

  edm::ConsumesCollector coll = consumesCollector();
  const std::vector<edm::ParameterSet>& importers = iConfig.getParameterSetVector("elementImporters");
  pfBlockAlgo_.setImporters(importers, coll);

  const std::vector<edm::ParameterSet>& linkdefs = iConfig.getParameterSetVector("linkDefinitions");
  pfBlockAlgo_.setLinkers(linkdefs);
}

PFBlockProducer::~PFBlockProducer() {}

void PFBlockProducer::beginLuminosityBlock(edm::LuminosityBlock const& lb, edm::EventSetup const& es) {
  pfBlockAlgo_.updateEventSetup(es);
}

void PFBlockProducer::produce(Event& iEvent, const EventSetup& iSetup) {
  pfBlockAlgo_.buildElements(iEvent);

  auto blocks = pfBlockAlgo_.findBlocks();

  if (verbose_) {
    ostringstream str;
    str << pfBlockAlgo_ << endl;
    str << "number of blocks : " << blocks.size() << endl;
    str << endl;

    for (PFBlockAlgo::IBC ib = blocks.begin(); ib != blocks.end(); ++ib) {
      str << (*ib) << endl;
    }

    LogInfo("PFBlockProducer") << str.str() << endl;
  }

  iEvent.emplace(putToken_, blocks);
}

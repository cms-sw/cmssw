#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <array>
#include <iterator>
#include <sstream>
#include <type_traits>
#include <utility>

#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "RecoParticleFlow/PFProducer/interface/BlockElementImporterBase.h"
#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
#include "RecoParticleFlow/PFProducer/interface/KDTreeLinkerBase.h"
#include "RecoParticleFlow/PFProducer/interface/PFBlockLink.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ProductRegistryHelper.h"
#include "FWCore/Framework/src/WorkerMaker.h"
#include "FWCore/MessageLogger/interface/ErrorObj.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFiller.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

namespace edm {
class EventSetup;
class LuminosityBlock;
}  // namespace edm

namespace std {
  template <> struct hash<std::pair<unsigned int, unsigned int> > {
    typedef std::pair<unsigned int, unsigned int> arg_type;
    typedef unsigned int value_type;
    value_type operator()(const arg_type& arg) const { return arg.first ^ (arg.second << 1); }
  };
}  // namespace std

/**\class PFBlockProducer
\brief Producer for particle flow blocks

This producer makes use of PFBlockProducer, the particle flow block algorithm.
Particle flow itself consists in reconstructing particles from the particle
flow blocks This is done at a later stage, see PFProducer and PFAlgo.

\author Colin Bernet
\date   April 2007
*/

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

  void setLinkers(const std::vector<edm::ParameterSet>&);

  void setImporters(const std::vector<edm::ParameterSet>&, edm::ConsumesCollector&);

  // run all of the importers and build KDtrees
  void buildElements(const edm::Event&);

  /// build blocks
  reco::PFBlockCollection findBlocks();

  /// sets debug printout flag
  void setDebug(bool debug) { debug_ = debug; }

  /// compute missing links in the blocks
  /// (the recursive procedure does not build all links)
  void packLinks(reco::PFBlock& block,
                 const std::unordered_map<std::pair<unsigned int, unsigned int>, PFBlockLink>& links) const;

  /// Avoid to check links when not useful
  inline bool linkPrefilter(const reco::PFBlockElement* last, const reco::PFBlockElement* next) const;

  /// check whether 2 elements are linked. Returns distance and linktype
  inline double link(const reco::PFBlockElement* el1,
                   const reco::PFBlockElement* el2,
                   PFBlockLink::Type& linktype,
                   reco::PFBlock::LinkTest& linktest) const;

  // the test elements will be transferred to the blocks
  std::vector<std::unique_ptr<reco::PFBlockElement>> elements_;
  std::array<std::pair<unsigned int, unsigned int>, reco::PFBlockElement::kNBETypes> ranges_;

  /// if true, debug printouts activated
  bool debug_;

  bool useHO_;

  std::vector<std::unique_ptr<BlockElementImporterBase>> importers_;

  const std::unordered_map<std::string, reco::PFBlockElement::Type> elementTypes_;
  std::vector<std::unique_ptr<BlockElementLinkerBase>> linkTests_;
  unsigned int linkTestSquare_[reco::PFBlockElement::kNBETypes][reco::PFBlockElement::kNBETypes];

  std::vector<std::unique_ptr<KDTreeLinkerBase>> kdtrees_;

};

DEFINE_FWK_MODULE(PFBlockProducer);


using namespace std;
using namespace reco;
using namespace edm;


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
} // namespace


void PFBlockProducer::setLinkers(const std::vector<edm::ParameterSet>& confs) {
  for (unsigned i = 0; i < reco::PFBlockElement::kNBETypes; ++i) {
    for (unsigned j = 0; j < reco::PFBlockElement::kNBETypes; ++j) {
      linkTestSquare_[i][j] = 0;
    }
  }
  linkTests_.resize(reco::PFBlockElement::kNBETypes * reco::PFBlockElement::kNBETypes);
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
    const unsigned index = reco::PFBlockElement::kNBETypes * std::max(type1, type2) + std::min(type1, type2);
    linkTests_[index] = std::unique_ptr<BlockElementLinkerBase>{BlockElementLinkerFactory::get()->create(linkerName, conf)};
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

void PFBlockProducer::setImporters(const std::vector<edm::ParameterSet>& confs, edm::ConsumesCollector& sumes) {
  importers_.reserve(confs.size());
  for (const auto& conf : confs) {
    const std::string& importerName = conf.getParameter<std::string>("importerName");
    importers_.emplace_back(BlockElementImporterFactory::get()->create(importerName, conf, sumes));
  }
}

PFBlockProducer::~PFBlockProducer() {
  if (debug_)
    cout << "~PFBlockProducer - number of remaining elements: " << elements_.size() << endl;
}

reco::PFBlockCollection PFBlockProducer::findBlocks() {
  // Glowinski & Gouzevitch
  for (const auto& kdtree : kdtrees_) {
    kdtree->process();
  }
  // !Glowinski & Gouzevitch
  reco::PFBlockCollection blocks;
  // the blocks have not been passed to the event, and need to be cleared
  blocks.reserve(elements_.size());

  QuickUnion quickUnion(elements_.size());
  const auto elem_size = elements_.size();
  for (unsigned i = 0; i < elem_size; ++i) {
    for (unsigned j = 0; j < elem_size; ++j) {
      if (quickUnion.connected(i, j) || j == i)
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
          quickUnion.unite(i, j);
        }
      }
    }
  }

  std::unordered_multimap<unsigned, unsigned> blocksmap(elements_.size());
  std::vector<unsigned> keys;
  keys.reserve(elements_.size());
  for (unsigned i = 0; i < elements_.size(); ++i) {
    unsigned key = i;
    while (key != quickUnion.find(key))
      key = quickUnion.find(key);  // make sure we always find the root node...
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
    reco::PFBlockElement* p1(elements_[range.first->second].get());
    the_block.addElement(p1);
    const unsigned block_size = blocksmap.count(key) + 1;
    //reserve up to 1M or 8MB; pay rehash cost for more
    std::unordered_map<std::pair<unsigned int, unsigned int>, PFBlockLink> links(
        min(1000000u, block_size * block_size));
    for (auto itr = std::next(range.first); itr != range.second; ++itr) {
      reco::PFBlockElement* p2(elements_[itr->second].get());
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

void PFBlockProducer::packLinks(reco::PFBlock& block,
                            const std::unordered_map<std::pair<unsigned int, unsigned int>, PFBlockLink>& links) const {

  const edm::OwnVector<reco::PFBlockElement>& els = block.elements();

  block.bookLinkData();
  unsigned elsize = els.size();
  //First Loop: update all link data
  for (unsigned i1 = 0; i1 < elsize; ++i1) {
    for (unsigned i2 = 0; i2 < i1; ++i2) {

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
        const unsigned index = reco::PFBlockElement::kNBETypes * minmax.second + minmax.first;
        PFBlockLink::Type linktype = PFBlockLink::NONE;
        bool bTestLink =
            (nullptr == linkTests_[index] ? false : linkTests_[index]->linkPrefilter(&(els[i1]), &(els[i2])));
        if (bTestLink)
          dist = link(&els[i1], &els[i2], linktype, linktest);
      }

      //loading link data according to link test used: RECHIT
      //block.setLink( i1, i2, chi2, block.linkData() );
      if (debug_)
        cout << "Setting link between elements " << i1 << " and " << i2 << " of dist =" << dist
             << " computed from link test " << linktest << endl;
      block.setLink(i1, i2, dist, block.linkData(), linktest);
    }
  }
}

// see plugins/linkers for the functions that calculate distances
// for each available link type
inline bool PFBlockProducer::linkPrefilter(const reco::PFBlockElement* last, const reco::PFBlockElement* next) const {
  const PFBlockElement::Type type1 = (last)->type();
  const PFBlockElement::Type type2 = (next)->type();
  const unsigned index = reco::PFBlockElement::kNBETypes * std::max(type1, type2) + std::min(type1, type2);
  bool result = linkTests_[index]->linkPrefilter(last, next);
  return result;
}

inline double PFBlockProducer::link(const reco::PFBlockElement* el1,
                              const reco::PFBlockElement* el2,
                              PFBlockLink::Type& linktype,
                              reco::PFBlock::LinkTest& linktest) const {
  double dist = -1.0;
  linktest = PFBlock::LINKTEST_RECHIT;  //rechit by default
  const PFBlockElement::Type type1 = el1->type();
  const PFBlockElement::Type type2 = el2->type();
  linktype = static_cast<PFBlockLink::Type>(1 << (type1 - 1) | 1 << (type2 - 1));
  const unsigned index = reco::PFBlockElement::kNBETypes * std::max(type1, type2) + std::min(type1, type2);
  if (debug_) {
    std::cout << " PFBlockProducer links type1 " << type1 << " type2 " << type2 << std::endl;
  }

  // index is always checked in the preFilter above, no need to check here
  dist = linkTests_[index]->testLink(el1, el2);
  return dist;
}

// see plugins/importers and plugins/kdtrees
// for the definitions of available block element importers
// and kdtree preprocessors
void PFBlockProducer::buildElements(const edm::Event& evt) {
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

  for(auto const& element : elements_) {
    for (const auto& kdtree : kdtrees_) {
      if (element->type() == kdtree->targetType()) {
        kdtree->insertTargetElt(element.get());
      }
      if (element->type() == kdtree->fieldType()) {
        kdtree->insertFieldClusterElt(element.get());
      }
    }
  }
  //std::cout << "(new) imported: " << elements_.size() << " elements!" << std::endl;
}

PFBlockProducer::PFBlockProducer(const edm::ParameterSet& iConfig)
    : verbose_{iConfig.getUntrackedParameter<bool>("verbose", false)}
    , putToken_{produces<reco::PFBlockCollection>()}
    , debug_(false)
    , elementTypes_({{"PFBlockElement::TRACK", PFBlockElement::TRACK},
                     {"PFBlockElement::PS1", PFBlockElement::PS1},
                     {"PFBlockElement::PS2", PFBlockElement::PS2},
                     {"PFBlockElement::ECAL", PFBlockElement::ECAL},
                     {"PFBlockElement::HCAL", PFBlockElement::HCAL},
                     {"PFBlockElement::GSF", PFBlockElement::GSF},
                     {"PFBlockElement::BREM", PFBlockElement::BREM},
                     {"PFBlockElement::HFEM", PFBlockElement::HFEM},
                     {"PFBlockElement::HFHAD", PFBlockElement::HFHAD},
                     {"PFBlockElement::SC", PFBlockElement::SC},
                     {"PFBlockElement::HO", PFBlockElement::HO},
                     {"PFBlockElement::HGCAL", PFBlockElement::HGCAL}})
{
  bool debug_ = iConfig.getUntrackedParameter<bool>("debug", false);
  setDebug(debug_);

  edm::ConsumesCollector coll = consumesCollector();
  const std::vector<edm::ParameterSet>& importers = iConfig.getParameterSetVector("elementImporters");
  setImporters(importers, coll);

  const std::vector<edm::ParameterSet>& linkdefs = iConfig.getParameterSetVector("linkDefinitions");
  setLinkers(linkdefs);
}

void PFBlockProducer::beginLuminosityBlock(edm::LuminosityBlock const& lb, edm::EventSetup const& es) {
  // update event setup info of all linkers
  for (auto& importer : importers_) {
    importer->updateEventSetup(es);
  }
}

void PFBlockProducer::produce(Event& iEvent, const EventSetup& iSetup) {
  buildElements(iEvent);

  auto blocks = findBlocks();

  if (verbose_) {
    ostringstream str;
    str << "====== Particle Flow Block Algorithm ======= ";
    str << endl;
    str << "number of unassociated elements : " << elements_.size() << endl;
    str << endl;

    for(auto const& element : elements_) {
      str << "\t" << *element << endl;
    }
    str << "number of blocks : " << blocks.size() << endl;
    str << endl;

    for(auto const& block : blocks) {
      str << block << endl;
    }

    LogInfo("PFBlockProducer") << str.str() << endl;
  }

  iEvent.emplace(putToken_, blocks);
}

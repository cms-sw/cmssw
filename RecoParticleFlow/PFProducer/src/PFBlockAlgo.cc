#include "RecoParticleFlow/PFProducer/interface/PFBlockAlgo.h"
#include "FWCore/Framework/interface/ProductRegistryHelper.h"
#include "FWCore/Framework/src/WorkerMaker.h"
#include "FWCore/MessageLogger/interface/ErrorObj.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFiller.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

#include <algorithm>
#include <iostream>
#include <array>
#include <iterator>
#include <sstream>
#include <type_traits>
#include <utility>

//for debug only
//#define PFLOW_DEBUG

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

//this should be moved to importers
void makeTrackTables(size_t track_first, size_t track_last, const ElementList& elements, PFTables& tables) {
  std::set<std::pair<reco::PFBlockElement*, size_t>> tracks;
  for (size_t ielem = track_first; ielem <= track_last; ielem++) {
    reco::PFBlockElement* track = elements[ielem].get();
    // if (track->convRefs().size()>0)
    //   std::cout << "convrefs=" << track->convRefs().size() << std::endl;
    tracks.insert(std::make_pair(track, ielem));
  }
  std::vector<reco::PFBlockElement*> tracks_vec;
  std::vector<size_t> track_to_element;
  std::vector<size_t> element_to_track(elements.size(), std::numeric_limits<size_t>::max());

  for (const auto& elem : tracks) {
    size_t vec_elem = tracks_vec.size();
    track_to_element.push_back(elem.second);
    element_to_track[elem.second] = vec_elem;
    tracks_vec.push_back(elem.first);
  }
  tracks.clear();

  std::vector<ConversionRef> convrefs;
  std::vector<std::vector<size_t>> track_to_convrefs(tracks_vec.size(), std::vector<size_t>());
  size_t itrack = 0;
  size_t iconvref = 0;
  for (const auto* track : tracks_vec) {
    for (const auto& convref : track->convRefs()) {
      convrefs.push_back(convref);
      track_to_convrefs[itrack].push_back(iconvref);
      iconvref++;
    }
    itrack++;
  }

  tables.track_to_element = track_to_element;
  tables.element_to_track = element_to_track;
  tables.track_table_vertex = edm::soa::makeTrackTableVertex(tracks_vec);
  tables.track_to_convrefs = track_to_convrefs;
  tables.convref_table = edm::soa::makeConvRefTable(convrefs);
  tables.track_table_ecalshowermax = edm::soa::makeTrackTable(tracks_vec, reco::PFTrajectoryPoint::ECALShowerMax);
  tables.track_table_hcalent = edm::soa::makeTrackTable(tracks_vec, reco::PFTrajectoryPoint::HCALEntrance);
  tables.track_table_hcalex = edm::soa::makeTrackTable(tracks_vec, reco::PFTrajectoryPoint::HCALExit);
  tables.track_table_vfcalent = edm::soa::makeTrackTable(tracks_vec, reco::PFTrajectoryPoint::VFcalEntrance);
  tables.track_table_ho = edm::soa::makeTrackTable(tracks_vec, reco::PFTrajectoryPoint::HOLayer);
}

//this should be moved to importers
template <class ClusterType>
PFClusterTables<edm::soa::ClusterTable, edm::soa::RecHitTable> makeClusterTables(size_t elem_first,
                                                                                 size_t elem_last,
                                                                                 const ElementList& elements,
                                                                                 float cutOffFrac) {
  //use sets to keep only unique elements
  std::set<std::pair<const ClusterType*, size_t>> clusters;
  std::set<const reco::PFRecHitFraction*> rechits;
  std::unordered_map<const reco::PFRecHitFraction*, std::set<const ClusterType*>> rechit2clusters;
  for (size_t ielem = elem_first; ielem <= elem_last; ielem++) {
    const ClusterType* cluster = static_cast<const ClusterType*>(elements[ielem].get());
    clusters.insert(std::make_pair(cluster, ielem));

    if (cluster->clusterRef().isNull()) {
      throw cms::Exception("BadRef") << "Cluster ref for element is invalid";
    }
    for (const auto& rechit : elements[ielem]->clusterRef()->recHitFractions()) {
      const reco::PFRecHitRef& rh = rechit.recHitRef();
      const double fract = rechit.fraction();

      if ((rh.isNull()) || (fract < cutOffFrac))
        continue;

      rechits.insert(&rechit);
      rechit2clusters[&rechit].insert(cluster);
    }
  }

  //convert sets to vectors that can be indexed, keeping the same order
  std::vector<const ClusterType*> clusters_vec;
  std::vector<size_t> cluster_to_element;
  std::vector<size_t> element_to_cluster(elements.size(), std::numeric_limits<size_t>::max());
  for (const auto& cluster_and_elemidx : clusters) {
    size_t cluster_idx = clusters_vec.size();
    cluster_to_element.push_back(cluster_and_elemidx.second);
    element_to_cluster[cluster_and_elemidx.second] = cluster_idx;
    //std::cout << "cluster eta=" << cluster_and_elemidx.first->clusterRef()->positionREP().eta()
    //  << " phi=" << cluster_and_elemidx.first->clusterRef()->positionREP().phi()
    //  << " idx=" << cluster_idx << std::endl;
    clusters_vec.push_back(cluster_and_elemidx.first);
  }
  clusters.clear();

  std::vector<const reco::PFRecHitFraction*> rechits_vec;
  rechits_vec.insert(rechits_vec.end(), rechits.begin(), rechits.end());
  rechits.clear();

  //convert pointer-based map to map of indices
  std::unordered_map<size_t, std::set<size_t>> rechit2clustersIdx;
  std::unordered_map<size_t, std::set<size_t>> cluster_to_rechit;
  for (const auto& rh_cluster : rechit2clusters) {
    const auto* rechit = rh_cluster.first;
    const auto idx_rechit =
        std::distance(rechits_vec.begin(), std::find(rechits_vec.begin(), rechits_vec.end(), rechit));
    for (const auto* cluster : rh_cluster.second) {
      const auto idx_cluster =
          std::distance(clusters_vec.begin(), std::find(clusters_vec.begin(), clusters_vec.end(), cluster));
      rechit2clustersIdx[idx_rechit].insert(idx_cluster);
      cluster_to_rechit[idx_cluster].insert(idx_rechit);
    }
  }
  rechit2clusters.clear();

  return {std::move(edm::soa::makeClusterTable(clusters_vec)),
          std::move(edm::soa::makeRecHitTable(rechits_vec)),
          std::move(cluster_to_element),
          std::move(element_to_cluster),
          std::move(rechit2clustersIdx),
          std::move(cluster_to_rechit)};
}

PFClusterTables<edm::soa::SuperClusterTable, edm::soa::SuperClusterRecHitTable> makeSuperClusterTables(
    size_t elem_first, size_t elem_last, const ElementList& elements, float cutOffFrac) {
  using key_type = const std::pair<DetId, float>*;
  //use sets to keep only unique elements
  std::set<std::pair<const PFBlockElementSuperCluster*, size_t>> clusters;
  std::set<key_type> rechits;
  std::unordered_map<key_type, std::set<const PFBlockElementSuperCluster*>> rechit2clusters;
  for (size_t ielem = elem_first; ielem <= elem_last; ielem++) {
    const PFBlockElementSuperCluster* cluster = static_cast<const PFBlockElementSuperCluster*>(elements[ielem].get());
    if (cluster->superClusterRef().isNull()) {
      throw cms::Exception("BadRef") << "SuperCluster ref for element is invalid";
    }
    clusters.insert(std::make_pair(cluster, ielem));

    //check is needed for SC
    for (const auto& rechit : cluster->superClusterRef()->hitsAndFractions()) {
      rechits.insert(&rechit);
      rechit2clusters[&rechit].insert(cluster);
    }
  }

  //convert sets to vectors that can be indexed, keeping the same order
  std::vector<const PFBlockElementSuperCluster*> clusters_vec;
  std::vector<size_t> cluster_to_element;
  std::vector<size_t> element_to_cluster(elements.size(), std::numeric_limits<size_t>::max());
  for (const auto& cluster_and_elemidx : clusters) {
    size_t cluster_idx = clusters_vec.size();
    cluster_to_element.push_back(cluster_and_elemidx.second);
    element_to_cluster[cluster_and_elemidx.second] = cluster_idx;
    clusters_vec.push_back(cluster_and_elemidx.first);
  }
  clusters.clear();

  std::vector<key_type> rechits_vec;
  rechits_vec.insert(rechits_vec.end(), rechits.begin(), rechits.end());
  rechits.clear();

  //convert pointer-based map to map of indices
  std::unordered_map<size_t, std::set<size_t>> rechit2clustersIdx;
  std::unordered_map<size_t, std::set<size_t>> cluster_to_rechit;
  for (const auto& rh_cluster : rechit2clusters) {
    const auto* rechit = rh_cluster.first;
    const auto idx_rechit =
        std::distance(rechits_vec.begin(), std::find(rechits_vec.begin(), rechits_vec.end(), rechit));
    for (const auto* cluster : rh_cluster.second) {
      const auto idx_cluster =
          std::distance(clusters_vec.begin(), std::find(clusters_vec.begin(), clusters_vec.end(), cluster));
      rechit2clustersIdx[idx_rechit].insert(idx_cluster);
      cluster_to_rechit[idx_cluster].insert(idx_rechit);
    }
  }
  rechit2clusters.clear();

  return {edm::soa::makeSuperClusterTable(clusters_vec),
          edm::soa::makeSuperClusterRecHitTable(rechits_vec),
          std::move(cluster_to_element),
          std::move(element_to_cluster),
          std::move(rechit2clustersIdx),
          std::move(cluster_to_rechit)};
}

void makeGSFTables(size_t idx_first, size_t idx_last, const ElementList& elements, PFTables& tables) {
  std::set<std::pair<const reco::PFBlockElementGsfTrack*, size_t>> tracks;
  for (size_t ielem = idx_first; ielem <= idx_last; ielem++) {
    const reco::PFBlockElementGsfTrack* track = static_cast<const PFBlockElementGsfTrack*>(elements[ielem].get());
    tracks.insert(std::make_pair(track, ielem));
  }
  std::vector<const reco::PFBlockElementGsfTrack*> tracks_vec;
  std::vector<size_t> track_to_element;
  std::vector<size_t> element_to_track(elements.size(), std::numeric_limits<size_t>::max());

  for (const auto& elem : tracks) {
    size_t vec_elem = tracks_vec.size();
    track_to_element.push_back(elem.second);
    element_to_track[elem.second] = vec_elem;
    tracks_vec.push_back(elem.first);
  }
  tracks.clear();

  std::vector<std::vector<size_t>> gsf_to_convbrem(tracks_vec.size(), std::vector<size_t>());
  std::vector<reco::PFRecTrackRef> convbrems;
  size_t itrack = 0;
  size_t iconvbrem = 0;
  for (const auto* track : tracks_vec) {
    for (const auto& convbrem : track->GsftrackRefPF()->convBremPFRecTrackRef()) {
      convbrems.push_back(convbrem);
      gsf_to_convbrem[itrack].push_back(iconvbrem);
      iconvbrem++;
    }
    itrack++;
  }

  tables.gsf_to_convbrem = gsf_to_convbrem;
  tables.gsf_table = edm::soa::makeGSFTable(tracks_vec);
  tables.gsf_table_ecalshowermax = edm::soa::makeTrackTable(tracks_vec, reco::PFTrajectoryPoint::ECALShowerMax);
  tables.gsf_table_hcalent = edm::soa::makeTrackTable(tracks_vec, reco::PFTrajectoryPoint::HCALEntrance);
  tables.gsf_table_hcalex = edm::soa::makeTrackTable(tracks_vec, reco::PFTrajectoryPoint::HCALExit);
  tables.gsf_convbrem_table = edm::soa::makeConvBremTable(convbrems);
  tables.gsf_to_element = track_to_element;
  tables.element_to_gsf = element_to_track;
}

void makeBREMTables(size_t idx_first, size_t idx_last, const ElementList& elements, PFTables& tables) {
  std::set<std::pair<const reco::PFBlockElementBrem*, size_t>> tracks;
  for (size_t ielem = idx_first; ielem <= idx_last; ielem++) {
    const reco::PFBlockElementBrem* track = static_cast<const PFBlockElementBrem*>(elements[ielem].get());
    tracks.insert(std::make_pair(track, ielem));
  }
  std::vector<const reco::PFBlockElementBrem*> tracks_vec;
  std::vector<size_t> track_to_element;
  std::vector<size_t> element_to_track(elements.size(), std::numeric_limits<size_t>::max());

  for (const auto& elem : tracks) {
    size_t vec_elem = tracks_vec.size();
    track_to_element.push_back(elem.second);
    element_to_track[elem.second] = vec_elem;
    tracks_vec.push_back(elem.first);
  }
  tracks.clear();

  tables.brem_table = edm::soa::makeBremTable(tracks_vec);
  tables.brem_table_ecalshowermax = edm::soa::makeTrackTable(tracks_vec, reco::PFTrajectoryPoint::ECALShowerMax);
  tables.brem_table_hcalent = edm::soa::makeTrackTable(tracks_vec, reco::PFTrajectoryPoint::HCALEntrance);
  tables.element_to_brem = element_to_track;
}

//Create a vector with constant pointers
const ElementListConst makeConstElements(const ElementList& elements) {
  ElementListConst elements_const;
  for (const auto& elem : elements) {
    elements_const.push_back(const_cast<const reco::PFBlockElement*>(elem.get()));
  }
  return elements_const;
}

//In the future, the tables can be read from the event when they are prepared by earlier reconstruction modules
void makeAllTables(const PFBlockAlgo::ElementRanges ranges,
                   const ElementList& elements,
                   PFTables& tables,
                   double cutOffFrac) {
  constexpr auto maxidx = std::numeric_limits<unsigned int>::max();

  //prepare track tables
  const auto& range_track = ranges.at(reco::PFBlockElement::TRACK);
  if (not(range_track.first == maxidx && range_track.second == 0)) {
    LogDebug("PFBlockAlgo") << "TRACK tables " << range_track.first << " " << range_track.second;
    makeTrackTables(range_track.first, range_track.second, elements, tables);
  }

  //prepare GSF tables
  const auto& range_gsf = ranges.at(reco::PFBlockElement::GSF);
  if (not(range_gsf.first == maxidx && range_gsf.second == 0)) {
    LogDebug("PFBlockAlgo") << "GSF tables" << range_gsf.first << " " << range_gsf.second;
    makeGSFTables(range_gsf.first, range_gsf.second, elements, tables);
  }

  //prepare BREM tables
  const auto& range_brem = ranges.at(reco::PFBlockElement::BREM);
  if (not(range_brem.first == maxidx && range_brem.second == 0)) {
    LogDebug("PFBlockAlgo") << "BREM tables " << range_brem.first << " " << range_brem.second;
    makeBREMTables(range_brem.first, range_brem.second, elements, tables);
  }

  //prepare calo cluster tables
  const auto& range_ecal = ranges.at(reco::PFBlockElement::ECAL);
  if (not(range_ecal.first == maxidx && range_ecal.second == 0)) {
    LogDebug("PFBlockAlgo") << "ECAL tables " << range_ecal.first << " " << range_ecal.second;
    tables.clusters_ecal =
        makeClusterTables<reco::PFBlockElementCluster>(range_ecal.first, range_ecal.second, elements, cutOffFrac);
  }

  const auto& range_hcal = ranges.at(reco::PFBlockElement::HCAL);
  if (not(range_hcal.first == maxidx && range_hcal.second == 0)) {
    LogDebug("PFBlockAlgo") << "HCAL tables " << range_hcal.first << " " << range_hcal.second;
    tables.clusters_hcal =
        makeClusterTables<reco::PFBlockElementCluster>(range_hcal.first, range_hcal.second, elements, cutOffFrac);
  }

  const auto& range_hfem = ranges.at(reco::PFBlockElement::HFEM);
  if (not(range_hfem.first == maxidx && range_hfem.second == 0)) {
    LogDebug("PFBlockAlgo") << "HFEM tables " << range_hfem.first << " " << range_hfem.second;
    tables.clusters_hfem =
        makeClusterTables<reco::PFBlockElementCluster>(range_hfem.first, range_hfem.second, elements, cutOffFrac);
  }

  const auto& range_hfhad = ranges.at(reco::PFBlockElement::HFHAD);
  if (not(range_hfhad.first == maxidx && range_hfhad.second == 0)) {
    LogDebug("PFBlockAlgo") << "HFHAD tables " << range_hfhad.first << " " << range_hfhad.second;
    tables.clusters_hfhad =
        makeClusterTables<reco::PFBlockElementCluster>(range_hfhad.first, range_hfhad.second, elements, cutOffFrac);
  }

  const auto& range_ps1 = ranges.at(reco::PFBlockElement::PS1);
  if (not(range_ps1.first == maxidx && range_ps1.second == 0)) {
    LogDebug("PFBlockAlgo") << "PS1 tables " << range_ps1.first << " " << range_ps1.second;
    tables.clusters_ps1 =
        makeClusterTables<reco::PFBlockElementCluster>(range_ps1.first, range_ps1.second, elements, cutOffFrac);
  }

  const auto& range_ps2 = ranges.at(reco::PFBlockElement::PS2);
  if (not(range_ps2.first == maxidx && range_ps2.second == 0)) {
    LogDebug("PFBlockAlgo") << "PS2 tables " << range_ps2.first << " " << range_ps2.second;
    tables.clusters_ps2 =
        makeClusterTables<reco::PFBlockElementCluster>(range_ps2.first, range_ps2.second, elements, cutOffFrac);
  }

  const auto& range_ho = ranges.at(reco::PFBlockElement::HO);
  if (not(range_ho.first == maxidx && range_ho.second == 0)) {
    LogDebug("PFBlockAlgo") << "HO tables " << range_ho.first << " " << range_ho.second;
    tables.clusters_ho =
        makeClusterTables<reco::PFBlockElementCluster>(range_ho.first, range_ho.second, elements, cutOffFrac);
  }

  const auto& range_sc = ranges.at(reco::PFBlockElement::SC);
  if (not(range_sc.first == maxidx && range_sc.second == 0)) {
    LogDebug("PFBlockAlgo") << "SC tables " << range_sc.first << " " << range_sc.second;
    tables.clusters_sc = makeSuperClusterTables(range_sc.first, range_sc.second, elements, cutOffFrac);
  }
}
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
    linkTests_[index] = BlockElementLinkerFactory::get()->create(linkerName, conf);

    linkTestSquare_[type1][type2] = index;
    linkTestSquare_[type2][type1] = index;
    // setup KDtree if requested
    const bool useKDTree = conf.getParameter<bool>("useKDTree");
    if (useKDTree) {
      kdtrees_.emplace_back(KDTreeLinkerFactory::get()->create(pfx_kdtree + linkerName, conf));
      kdtrees_.back()->setTargetType(std::min(type1, type2));
      kdtrees_.back()->setFieldType(std::max(type1, type2));
    }  // useKDTree
  }    // loop over confs
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

reco::PFBlockCollection PFBlockAlgo::findBlocks(const PFTables& tables) {
  const auto& elements_const = makeConstElements(elements_);

  reco::PFMultiLinksIndex multilinks(elements_.size());

  // Glowinski & Gouzevitch
  LogDebug("PFBlockAlgo") << "precomputing link data using KD-trees";
  for (const auto& kdtree : kdtrees_) {
    kdtree->process(tables, elements_, multilinks);
  }

#ifdef EDM_ML_DEBUG
  size_t num_ml_track_hcal = 0;
  size_t num_ml_track_hfem = 0;
  size_t num_ml_track_hfhad = 0;
  size_t num_ml_ecal_track = 0;
  size_t num_ml_ps1_ecal = 0;
  size_t num_ml_ps2_ecal = 0;
  for (size_t i = 0; i < elements_.size(); i++) {
    num_ml_track_hcal += multilinks.getNumLinks(i, reco::PFBlockElement::TRACK, reco::PFBlockElement::HCAL);
    num_ml_track_hfem += multilinks.getNumLinks(i, reco::PFBlockElement::TRACK, reco::PFBlockElement::HFEM);
    num_ml_track_hfhad += multilinks.getNumLinks(i, reco::PFBlockElement::TRACK, reco::PFBlockElement::HFHAD);
    num_ml_ecal_track += multilinks.getNumLinks(i, reco::PFBlockElement::ECAL, reco::PFBlockElement::TRACK);
    num_ml_ps1_ecal += multilinks.getNumLinks(i, reco::PFBlockElement::PS1, reco::PFBlockElement::ECAL);
    num_ml_ps2_ecal += multilinks.getNumLinks(i, reco::PFBlockElement::PS2, reco::PFBlockElement::ECAL);
  }

  LogDebug("PFBlockAlgo") << "number of multilinks HCAL->TRACK: " << num_ml_track_hcal;
  LogDebug("PFBlockAlgo") << "number of multilinks HFEM->TRACK: " << num_ml_track_hfem;
  LogDebug("PFBlockAlgo") << "number of multilinks HFHAD->TRACK: " << num_ml_track_hfhad;
  LogDebug("PFBlockAlgo") << "number of multilinks ECAL->TRACK: " << num_ml_ecal_track;
  LogDebug("PFBlockAlgo") << "number of multilinks PS1->ECAL: " << num_ml_ps1_ecal;
  LogDebug("PFBlockAlgo") << "number of multilinks PS2->ECAL: " << num_ml_ps2_ecal;
#endif

  // !Glowinski & Gouzevitch
  reco::PFBlockCollection blocks;
  // the blocks have not been passed to the event, and need to be cleared
  blocks.reserve(elements_.size());

  LogDebug("PFBlockAlgo") << "creating initial link structure";
  QuickUnion qu(elements_.size());
  const auto elem_size = elements_.size();
  for (unsigned i = 0; i < elem_size; ++i) {
    for (unsigned j = i + 1; j < elem_size; ++j) {
      if (qu.connected(i, j))
        continue;
      if (!linkTests_[linkTestSquare_[elements_[i]->type()][elements_[j]->type()]]) {
        j = ranges_[elements_[j]->type()].second;
        continue;
      }
      auto p1(elements_[i].get()), p2(elements_[j].get());
      const PFBlockElement::Type type1 = p1->type();
      const PFBlockElement::Type type2 = p2->type();
      const unsigned index = linkTestSquare_[type1][type2];
      const auto pref = linkTests_[index]->linkPrefilter(i, j, type1, type2, tables, multilinks);
      LogTrace("PFBlockAlgo") << "i=" << i << " j=" << j << " prefilter=" << pref << " " << type1 << " " << type2;
      if (pref) {
        const double dist = linkTests_[index]->testLink(i, j, type1, type2, elements_const, tables, multilinks);
        LogTrace("PFBlockAlgo") << "dist=" << dist << std::endl;
        // compute linking info if it is possible
        if (dist > -0.5) {
          qu.unite(i, j);
          LogTrace("PFBlockAlgo") << "uniting" << std::endl;
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
    LogTrace("PFBlockAlgo") << "inserting " << key << " " << i << std::endl;
  }

  LogTrace("PFBlockAlgo") << "finding additional links";
  for (auto key : keys) {
    auto range = blocksmap.equal_range(key);
    std::vector<size_t> block_element_indices;

    const size_t ielem1 = range.first->second;
    ElementList::value_type::pointer p1(elements_[range.first->second].get());
    block_element_indices.push_back(ielem1);
    const unsigned block_size = blocksmap.count(key) + 1;
    //reserve up to 1M or 8MB; pay rehash cost for more
    std::unordered_map<std::pair<unsigned int, unsigned int>, double> links(min(1000000u, block_size * block_size));
    auto itr = range.first;
    ++itr;
    for (; itr != range.second; ++itr) {
      const size_t ielem2 = itr->second;
      ElementList::value_type::pointer p2(elements_[ielem2].get());
      const PFBlockElement::Type type1 = elements_[ielem1]->type();
      const PFBlockElement::Type type2 = elements_[ielem2]->type();
      block_element_indices.push_back(ielem2);
      const unsigned index = linkTestSquare_[type1][type2];
      if (nullptr != linkTests_[index]) {
        const double dist =
            linkTests_[index]->testLink(ielem1, ielem2, type1, type2, elements_const, tables, multilinks);
        links.emplace(std::make_pair(p1->index(), p2->index()), dist);
      }
    }
    blocks.push_back(packLinks(tables, block_element_indices, links, elements_const, multilinks));
  }
  LogDebug("PFBlockAlgo") << "findBlocks done, produced " << blocks.size() << " blocks";

  elements_.clear();

  return blocks;
}

const reco::PFBlock PFBlockAlgo::packLinks(
    const PFTables& tables,
    const std::vector<size_t>& block_element_indices,
    const std::unordered_map<std::pair<unsigned int, unsigned int>, double>& links,
    const ElementListConst& elements_full,
    const reco::PFMultiLinksIndex& multilinks) const {
  constexpr unsigned rowsize = reco::PFBlockElement::kNBETypes;

  LogDebug("PFBlockAlgo") << "packLinks";
  reco::PFBlock block;

  for (size_t elem_index : block_element_indices) {
    block.addElement(elements_full[elem_index]);
  }

  const edm::OwnVector<reco::PFBlockElement>& els = block.elements();
  block.bookLinkData();
  unsigned elsize = els.size();

  //First Loop: update all link data
  for (unsigned i1 = 0; i1 < elsize; ++i1) {
    for (unsigned i2 = 0; i2 < i1; ++i2) {
      // no reflexive link
      //if( i1==i2 ) continue;

      //indices of the block elements in the full event element list
      const auto global_i1 = block_element_indices[i1];
      const auto global_i2 = block_element_indices[i2];

      double dist = -1;

      bool linked = false;

      // are these elements already linked ?
      // this can be optimized
      const auto link_itr = links.find(std::make_pair(i2, i1));
      if (link_itr != links.end()) {
        dist = link_itr->second;
        linked = true;
      }

      if (!linked) {
        const PFBlockElement::Type type1 = els[i1].type();
        const PFBlockElement::Type type2 = els[i2].type();
        const auto minmax = std::minmax(type1, type2);
        const unsigned index = rowsize * minmax.second + minmax.first;
        LogTrace("PFBlockAlgo") << "packLinks testLink i1=" << global_i1 << " i2=" << global_i2 << " type1=" << type1
                                << " type2=" << type2;
        bool bTestLink = (nullptr == linkTests_[index] ? false
                                                       : linkTests_[index]->linkPrefilter(
                                                             global_i1, global_i2, type1, type2, tables, multilinks));
        if (bTestLink)
          link(global_i1, global_i2, dist, elements_full, tables, multilinks);
      }

      //loading link data according to link test used: RECHIT
      //block.setLink( i1, i2, chi2, block.linkData() );
      block.setLink(i1, i2, dist, block.linkData());
    }
  }
  LogDebug("PFBlockAlgo") << "packLinks done";

  return block;
}

inline void PFBlockAlgo::link(size_t iel1,
                              size_t iel2,
                              double& dist,
                              const ElementListConst& elements_,
                              const PFTables& tables,
                              const reco::PFMultiLinksIndex& multilinks) const {
  constexpr unsigned rowsize = reco::PFBlockElement::kNBETypes;
  dist = -1.0;
  const PFBlockElement::Type type1 = elements_[iel1]->type();
  const PFBlockElement::Type type2 = elements_[iel2]->type();
  const unsigned index = rowsize * std::max(type1, type2) + std::min(type1, type2);
  LogTrace("PFBlockAlgo") << " PFBlockAlgo::link type1=" << type1 << " type2=" << type2;

  // index is always checked in the preFilter above, no need to check here
  dist = linkTests_[index]->testLink(iel1, iel2, type1, type2, elements_, tables, multilinks);
}

void PFBlockAlgo::updateEventSetup(const edm::EventSetup& es) {
  for (auto& importer : importers_) {
    importer->updateEventSetup(es);
  }
}

// see plugins/importers and plugins/kdtrees
// for the definitions of available block element importers
// and kdtree preprocessors
const PFTables PFBlockAlgo::buildElements(const edm::Event& evt) {
  // import block elements as defined in python configuration
  //the ranges are inclusive, meaning [start, end]
  //note that the pair (max, 0) is treated as a special default case of no elements
  ranges_.fill(std::make_pair(std::numeric_limits<unsigned int>::max(), 0));
  elements_.clear();
  PFTables tables;

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
    LogDebug("PFBlockAlgo") << "elem i=" << i << " type=" << the_type;
  }

  makeAllTables(ranges_, elements_, tables, cutOffFrac_);
  return tables;
}

std::ostream& operator<<(std::ostream& out, const PFBlockAlgo& a) {
  if (!out)
    return out;

  out << "====== Particle Flow Block Algorithm ======= ";
  out << endl;
  out << "number of unassociated elements : " << a.elements_.size() << endl;
  out << endl;

  for (auto const& element : a.elements_) {
    out << "\t" << *element << endl;
  }

  return out;
}

// a little history, ideas we may want to keep around for later
/*
  // Links between the two preshower layers are not used for now - disable
  case PFBlockLink::PS1andPS2:
    {
      PFClusterRef  ps1ref = lowEl->clusterRef();
      PFClusterRef  ps2ref = highEl->clusterRef();
      assert( !ps1ref.isNull() );
      assert( !ps2ref.isNull() );
      // PJ - 14-May-09 : A link by rechit is needed here !
      // dist = testPS1AndPS2( *ps1ref, *ps2ref );
      dist = -1.;
      linktest = PFBlock::LINKTEST_RECHIT;
      break;
    }
  case PFBlockLink::TRACKandPS1:
  case PFBlockLink::TRACKandPS2:
    {
      //cout<<"TRACKandPS"<<endl;
      PFRecTrackRef trackref = lowEl->trackRefPF();
      PFClusterRef  clusterref = highEl->clusterRef();
      assert( !trackref.isNull() );
      assert( !clusterref.isNull() );
      // PJ - 14-May-09 : A link by rechit is needed here !
      dist = testTrackAndPS( *trackref, *clusterref );
      linktest = PFBlock::LINKTEST_RECHIT;
      break;
    }
    // GSF Track/Brem Track and preshower cluster links are not used for now - disable
  case PFBlockLink::PS1andGSF:
  case PFBlockLink::PS2andGSF:
    {
      PFClusterRef  psref = lowEl->clusterRef();
      assert( !psref.isNull() );
      const reco::PFBlockElementGsfTrack *  GsfEl =  dynamic_cast<const reco::PFBlockElementGsfTrack*>(highEl);
      const PFRecTrack * myTrack =  &(GsfEl->GsftrackPF());
      // PJ - 14-May-09 : A link by rechit is needed here !
      dist = testTrackAndPS( *myTrack, *psref );
      linktest = PFBlock::LINKTEST_RECHIT;
      break;
    }
  case PFBlockLink::PS1andBREM:
  case PFBlockLink::PS2andBREM:
    {
      PFClusterRef  psref = lowEl->clusterRef();
      assert( !psref.isNull() );
      const reco::PFBlockElementBrem * BremEl =  dynamic_cast<const reco::PFBlockElementBrem*>(highEl);
      const PFRecTrack * myTrack = &(BremEl->trackPF());
      // PJ - 14-May-09 : A link by rechit is needed here !
      dist = testTrackAndPS( *myTrack, *psref );
      linktest = PFBlock::LINKTEST_RECHIT;
      break;
    }
    */

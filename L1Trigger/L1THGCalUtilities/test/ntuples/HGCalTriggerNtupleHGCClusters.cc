
#include <algorithm>
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCalUtilities/interface/HGCalTriggerNtupleBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"

class HGCalTriggerNtupleHGCClusters : public HGCalTriggerNtupleBase {
public:
  HGCalTriggerNtupleHGCClusters(const edm::ParameterSet& conf);
  ~HGCalTriggerNtupleHGCClusters() override{};
  void initialize(TTree&, const edm::ParameterSet&, edm::ConsumesCollector&&) final;
  void fill(const edm::Event& e, const HGCalTriggerNtupleEventSetup& es) final;

private:
  void clear() final;

  bool filter_clusters_in_multiclusters_;
  edm::EDGetToken clusters_token_, multiclusters_token_;
  HGCalTriggerTools triggerTools_;

  int cl_n_;
  std::vector<uint32_t> cl_id_;
  std::vector<float> cl_mipPt_;
  std::vector<float> cl_pt_;
  std::vector<float> cl_energy_;
  std::vector<float> cl_eta_;
  std::vector<float> cl_phi_;
  std::vector<int> cl_layer_;
  std::vector<int> cl_subdet_;
  std::vector<int> cl_cells_n_;
  std::vector<std::vector<uint32_t>> cl_cells_id_;
  std::vector<uint32_t> cl_multicluster_id_;
  std::vector<float> cl_multicluster_pt_;
};

DEFINE_EDM_PLUGIN(HGCalTriggerNtupleFactory, HGCalTriggerNtupleHGCClusters, "HGCalTriggerNtupleHGCClusters");

HGCalTriggerNtupleHGCClusters::HGCalTriggerNtupleHGCClusters(const edm::ParameterSet& conf)
    : HGCalTriggerNtupleBase(conf),
      filter_clusters_in_multiclusters_(conf.getParameter<bool>("FilterClustersInMulticlusters")) {
  accessEventSetup_ = false;
}

void HGCalTriggerNtupleHGCClusters::initialize(TTree& tree,
                                               const edm::ParameterSet& conf,
                                               edm::ConsumesCollector&& collector) {
  clusters_token_ = collector.consumes<l1t::HGCalClusterBxCollection>(conf.getParameter<edm::InputTag>("Clusters"));
  multiclusters_token_ =
      collector.consumes<l1t::HGCalMulticlusterBxCollection>(conf.getParameter<edm::InputTag>("Multiclusters"));

  std::string prefix(conf.getUntrackedParameter<std::string>("Prefix", "cl"));

  std::string bname;
  auto withPrefix([&prefix, &bname](char const* vname) -> char const* {
    bname = prefix + "_" + vname;
    return bname.c_str();
  });

  // note: can't use withPrefix() twice within a same statement because bname gets overwritten
  tree.Branch(withPrefix("n"), &cl_n_, (prefix + "_n/I").c_str());
  tree.Branch(withPrefix("id"), &cl_id_);
  tree.Branch(withPrefix("mipPt"), &cl_mipPt_);
  tree.Branch(withPrefix("pt"), &cl_pt_);
  tree.Branch(withPrefix("energy"), &cl_energy_);
  tree.Branch(withPrefix("eta"), &cl_eta_);
  tree.Branch(withPrefix("phi"), &cl_phi_);
  tree.Branch(withPrefix("layer"), &cl_layer_);
  tree.Branch(withPrefix("subdet"), &cl_subdet_);
  tree.Branch(withPrefix("cells_n"), &cl_cells_n_);
  tree.Branch(withPrefix("cells_id"), &cl_cells_id_);
  tree.Branch(withPrefix("multicluster_id"), &cl_multicluster_id_);
  tree.Branch(withPrefix("multicluster_pt"), &cl_multicluster_pt_);
}

void HGCalTriggerNtupleHGCClusters::fill(const edm::Event& e, const HGCalTriggerNtupleEventSetup& es) {
  // retrieve clusters
  edm::Handle<l1t::HGCalClusterBxCollection> clusters_h;
  e.getByToken(clusters_token_, clusters_h);
  const l1t::HGCalClusterBxCollection& clusters = *clusters_h;
  edm::Handle<l1t::HGCalMulticlusterBxCollection> multiclusters_h;
  e.getByToken(multiclusters_token_, multiclusters_h);
  const l1t::HGCalMulticlusterBxCollection& multiclusters = *multiclusters_h;

  triggerTools_.setGeometry(es.geometry.product());

  // Associate cells to clusters
  std::unordered_map<uint32_t, l1t::HGCalMulticlusterBxCollection::const_iterator> cluster2multicluster;
  for (auto mcl_itr = multiclusters.begin(0); mcl_itr != multiclusters.end(0); mcl_itr++) {
    // loop on 2D clusters inside 3D clusters
    for (const auto& cl_ptr : mcl_itr->constituents()) {
      cluster2multicluster.emplace(cl_ptr.second->detId(), mcl_itr);
    }
  }

  clear();
  for (auto cl_itr = clusters.begin(0); cl_itr != clusters.end(0); cl_itr++) {
    auto mcl_itr = cluster2multicluster.find(cl_itr->detId());
    uint32_t mcl_id = (mcl_itr != cluster2multicluster.end() ? mcl_itr->second->detId() : 0);
    float mcl_pt = (mcl_itr != cluster2multicluster.end() ? mcl_itr->second->pt() : 0.);
    if (filter_clusters_in_multiclusters_ && mcl_id == 0)
      continue;
    cl_n_++;
    cl_mipPt_.emplace_back(cl_itr->mipPt());
    // physical values
    cl_pt_.emplace_back(cl_itr->pt());
    cl_energy_.emplace_back(cl_itr->energy());
    cl_eta_.emplace_back(cl_itr->eta());
    cl_phi_.emplace_back(cl_itr->phi());

    cl_id_.emplace_back(cl_itr->detId());
    cl_layer_.emplace_back(triggerTools_.layerWithOffset(cl_itr->detId()));
    cl_subdet_.emplace_back(cl_itr->subdetId());
    cl_cells_n_.emplace_back(cl_itr->constituents().size());
    // Retrieve indices of trigger cells inside cluster
    cl_cells_id_.emplace_back(cl_itr->constituents().size());
    std::transform(
        cl_itr->constituents_begin(),
        cl_itr->constituents_end(),
        cl_cells_id_.back().begin(),
        [](const std::pair<uint32_t, edm::Ptr<l1t::HGCalTriggerCell>>& id_tc) { return id_tc.second->detId(); });
    cl_multicluster_id_.emplace_back(mcl_id);
    cl_multicluster_pt_.emplace_back(mcl_pt);
  }
}

void HGCalTriggerNtupleHGCClusters::clear() {
  cl_n_ = 0;
  cl_id_.clear();
  cl_mipPt_.clear();
  cl_pt_.clear();
  cl_energy_.clear();
  cl_eta_.clear();
  cl_phi_.clear();
  cl_layer_.clear();
  cl_subdet_.clear();
  cl_cells_n_.clear();
  cl_cells_id_.clear();
  cl_multicluster_id_.clear();
  cl_multicluster_pt_.clear();
}

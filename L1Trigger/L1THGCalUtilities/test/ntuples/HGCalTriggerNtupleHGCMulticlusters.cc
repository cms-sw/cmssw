#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCalUtilities/interface/HGCalTriggerNtupleBase.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalTriggerClusterIdentificationBase.h"

class HGCalTriggerNtupleHGCMulticlusters : public HGCalTriggerNtupleBase {
public:
  HGCalTriggerNtupleHGCMulticlusters(const edm::ParameterSet& conf);
  ~HGCalTriggerNtupleHGCMulticlusters() override{};
  void initialize(TTree&, const edm::ParameterSet&, edm::ConsumesCollector&&) final;
  void fill(const edm::Event& e, const HGCalTriggerNtupleEventSetup& es) final;

private:
  void clear() final;

  edm::EDGetToken multiclusters_token_;

  bool fill_layer_info_;
  bool fill_interpretation_info_;

  std::unique_ptr<HGCalTriggerClusterIdentificationBase> id_;

  int cl3d_n_;
  std::vector<uint32_t> cl3d_id_;
  std::vector<float> cl3d_pt_;
  std::vector<float> cl3d_energy_;
  std::vector<float> cl3d_eta_;
  std::vector<float> cl3d_phi_;
  std::vector<int> cl3d_clusters_n_;
  std::vector<std::vector<uint32_t>> cl3d_clusters_id_;
  std::vector<std::vector<float>> cl3d_layer_pt_;
  // cluster shower shapes
  std::vector<int> cl3d_showerlength_;
  std::vector<int> cl3d_coreshowerlength_;
  std::vector<int> cl3d_firstlayer_;
  std::vector<int> cl3d_maxlayer_;
  std::vector<float> cl3d_seetot_;
  std::vector<float> cl3d_seemax_;
  std::vector<float> cl3d_spptot_;
  std::vector<float> cl3d_sppmax_;
  std::vector<float> cl3d_szz_;
  std::vector<float> cl3d_srrtot_;
  std::vector<float> cl3d_srrmax_;
  std::vector<float> cl3d_srrmean_;
  std::vector<float> cl3d_emaxe_;
  std::vector<float> cl3d_hoe_;
  std::vector<float> cl3d_meanz_;
  std::vector<float> cl3d_layer10_;
  std::vector<float> cl3d_layer50_;
  std::vector<float> cl3d_layer90_;
  std::vector<float> cl3d_ntc67_;
  std::vector<float> cl3d_ntc90_;
  std::vector<float> cl3d_bdteg_;
  std::vector<int> cl3d_quality_;
  std::vector<std::vector<float>> cl3d_ipt_;
  std::vector<std::vector<float>> cl3d_ienergy_;
};

DEFINE_EDM_PLUGIN(HGCalTriggerNtupleFactory, HGCalTriggerNtupleHGCMulticlusters, "HGCalTriggerNtupleHGCMulticlusters");

HGCalTriggerNtupleHGCMulticlusters::HGCalTriggerNtupleHGCMulticlusters(const edm::ParameterSet& conf)
    : HGCalTriggerNtupleBase(conf),
      fill_layer_info_(conf.getParameter<bool>("FillLayerInfo")),
      fill_interpretation_info_(conf.getParameter<bool>("FillInterpretationInfo")) {
  accessEventSetup_ = false;
}

void HGCalTriggerNtupleHGCMulticlusters::initialize(TTree& tree,
                                                    const edm::ParameterSet& conf,
                                                    edm::ConsumesCollector&& collector) {
  multiclusters_token_ =
      collector.consumes<l1t::HGCalMulticlusterBxCollection>(conf.getParameter<edm::InputTag>("Multiclusters"));
  id_ = std::unique_ptr<HGCalTriggerClusterIdentificationBase>{
      HGCalTriggerClusterIdentificationFactory::get()->create("HGCalTriggerClusterIdentificationBDT")};
  id_->initialize(conf.getParameter<edm::ParameterSet>("EGIdentification"));

  std::string prefix(conf.getUntrackedParameter<std::string>("Prefix", "cl3d"));

  std::string bname;
  auto withPrefix([&prefix, &bname](char const* vname) -> char const* {
    bname = prefix + "_" + vname;
    return bname.c_str();
  });

  tree.Branch(withPrefix("n"), &cl3d_n_, (prefix + "_n/I").c_str());
  tree.Branch(withPrefix("id"), &cl3d_id_);
  tree.Branch(withPrefix("pt"), &cl3d_pt_);
  tree.Branch(withPrefix("energy"), &cl3d_energy_);
  tree.Branch(withPrefix("eta"), &cl3d_eta_);
  tree.Branch(withPrefix("phi"), &cl3d_phi_);
  tree.Branch(withPrefix("clusters_n"), &cl3d_clusters_n_);
  tree.Branch(withPrefix("clusters_id"), &cl3d_clusters_id_);
  if (fill_layer_info_)
    tree.Branch(withPrefix("layer_pt"), &cl3d_layer_pt_);
  tree.Branch(withPrefix("showerlength"), &cl3d_showerlength_);
  tree.Branch(withPrefix("coreshowerlength"), &cl3d_coreshowerlength_);
  tree.Branch(withPrefix("firstlayer"), &cl3d_firstlayer_);
  tree.Branch(withPrefix("maxlayer"), &cl3d_maxlayer_);
  tree.Branch(withPrefix("seetot"), &cl3d_seetot_);
  tree.Branch(withPrefix("seemax"), &cl3d_seemax_);
  tree.Branch(withPrefix("spptot"), &cl3d_spptot_);
  tree.Branch(withPrefix("sppmax"), &cl3d_sppmax_);
  tree.Branch(withPrefix("szz"), &cl3d_szz_);
  tree.Branch(withPrefix("srrtot"), &cl3d_srrtot_);
  tree.Branch(withPrefix("srrmax"), &cl3d_srrmax_);
  tree.Branch(withPrefix("srrmean"), &cl3d_srrmean_);
  tree.Branch(withPrefix("emaxe"), &cl3d_emaxe_);
  tree.Branch(withPrefix("hoe"), &cl3d_hoe_);
  tree.Branch(withPrefix("meanz"), &cl3d_meanz_);
  tree.Branch(withPrefix("layer10"), &cl3d_layer10_);
  tree.Branch(withPrefix("layer50"), &cl3d_layer50_);
  tree.Branch(withPrefix("layer90"), &cl3d_layer90_);
  tree.Branch(withPrefix("ntc67"), &cl3d_ntc67_);
  tree.Branch(withPrefix("ntc90"), &cl3d_ntc90_);
  tree.Branch(withPrefix("bdteg"), &cl3d_bdteg_);
  tree.Branch(withPrefix("quality"), &cl3d_quality_);
  if (fill_interpretation_info_) {
    tree.Branch(withPrefix("ipt"), &cl3d_ipt_);
    tree.Branch(withPrefix("ienergy"), &cl3d_ienergy_);
  }
}

void HGCalTriggerNtupleHGCMulticlusters::fill(const edm::Event& e, const HGCalTriggerNtupleEventSetup& es) {
  // retrieve clusters 3D
  edm::Handle<l1t::HGCalMulticlusterBxCollection> multiclusters_h;
  e.getByToken(multiclusters_token_, multiclusters_h);
  const l1t::HGCalMulticlusterBxCollection& multiclusters = *multiclusters_h;

  clear();
  for (auto cl3d_itr = multiclusters.begin(0); cl3d_itr != multiclusters.end(0); cl3d_itr++) {
    cl3d_n_++;
    cl3d_id_.emplace_back(cl3d_itr->detId());
    // physical values
    cl3d_pt_.emplace_back(cl3d_itr->pt());
    cl3d_energy_.emplace_back(cl3d_itr->energy());
    cl3d_eta_.emplace_back(cl3d_itr->eta());
    cl3d_phi_.emplace_back(cl3d_itr->phi());
    cl3d_clusters_n_.emplace_back(cl3d_itr->constituents().size());
    cl3d_showerlength_.emplace_back(cl3d_itr->showerLength());
    cl3d_coreshowerlength_.emplace_back(cl3d_itr->coreShowerLength());
    cl3d_firstlayer_.emplace_back(cl3d_itr->firstLayer());
    cl3d_maxlayer_.emplace_back(cl3d_itr->maxLayer());
    cl3d_seetot_.emplace_back(cl3d_itr->sigmaEtaEtaTot());
    cl3d_seemax_.emplace_back(cl3d_itr->sigmaEtaEtaMax());
    cl3d_spptot_.emplace_back(cl3d_itr->sigmaPhiPhiTot());
    cl3d_sppmax_.emplace_back(cl3d_itr->sigmaPhiPhiMax());
    cl3d_szz_.emplace_back(cl3d_itr->sigmaZZ());
    cl3d_srrtot_.emplace_back(cl3d_itr->sigmaRRTot());
    cl3d_srrmax_.emplace_back(cl3d_itr->sigmaRRMax());
    cl3d_srrmean_.emplace_back(cl3d_itr->sigmaRRMean());
    cl3d_emaxe_.emplace_back(cl3d_itr->eMax() / cl3d_itr->energy());
    cl3d_hoe_.emplace_back(cl3d_itr->hOverE());
    cl3d_meanz_.emplace_back(std::abs(cl3d_itr->zBarycenter()));
    cl3d_layer10_.emplace_back(cl3d_itr->layer10percent());
    cl3d_layer50_.emplace_back(cl3d_itr->layer50percent());
    cl3d_layer90_.emplace_back(cl3d_itr->layer90percent());
    cl3d_ntc67_.emplace_back(cl3d_itr->triggerCells67percent());
    cl3d_ntc90_.emplace_back(cl3d_itr->triggerCells90percent());
    cl3d_bdteg_.emplace_back(id_->value(*cl3d_itr));
    cl3d_quality_.emplace_back(cl3d_itr->hwQual());
    if (fill_interpretation_info_) {
      std::vector<float> iPts(cl3d_itr->interpretations_size());
      std::vector<float> iEnergies(cl3d_itr->interpretations_size());
      for (auto interp = cl3d_itr->interpretations_begin(); interp != cl3d_itr->interpretations_end(); ++interp) {
        iPts.emplace_back(cl3d_itr->iPt(*interp));
        iEnergies.emplace_back(cl3d_itr->iEnergy(*interp));
      }
      cl3d_ipt_.push_back(iPts);
      cl3d_ienergy_.push_back(iEnergies);
    }

    //Per layer cluster information
    if (fill_layer_info_) {
      const unsigned nlayers = es.geometry->lastTriggerLayer();
      std::vector<float> layer_pt(nlayers, 0.0);
      for (const auto& cl_ptr : cl3d_itr->constituents()) {
        unsigned layer = es.geometry->triggerLayer(cl_ptr.second->detId());
        layer_pt[layer] += cl_ptr.second->pt();
      }
      cl3d_layer_pt_.emplace_back(layer_pt);
    }

    // Retrieve indices of trigger cells inside cluster
    cl3d_clusters_id_.emplace_back(cl3d_itr->constituents().size());
    std::transform(cl3d_itr->constituents_begin(),
                   cl3d_itr->constituents_end(),
                   cl3d_clusters_id_.back().begin(),
                   [](const std::pair<uint32_t, edm::Ptr<l1t::HGCalCluster>>& id_cl) { return id_cl.second->detId(); });
  }
}

void HGCalTriggerNtupleHGCMulticlusters::clear() {
  cl3d_n_ = 0;
  cl3d_id_.clear();
  cl3d_pt_.clear();
  cl3d_energy_.clear();
  cl3d_eta_.clear();
  cl3d_phi_.clear();
  cl3d_clusters_n_.clear();
  cl3d_clusters_id_.clear();
  cl3d_layer_pt_.clear();
  cl3d_showerlength_.clear();
  cl3d_coreshowerlength_.clear();
  cl3d_firstlayer_.clear();
  cl3d_maxlayer_.clear();
  cl3d_seetot_.clear();
  cl3d_seemax_.clear();
  cl3d_spptot_.clear();
  cl3d_sppmax_.clear();
  cl3d_szz_.clear();
  cl3d_srrtot_.clear();
  cl3d_srrmax_.clear();
  cl3d_srrmean_.clear();
  cl3d_emaxe_.clear();
  cl3d_hoe_.clear();
  cl3d_meanz_.clear();
  cl3d_layer10_.clear();
  cl3d_layer50_.clear();
  cl3d_layer90_.clear();
  cl3d_ntc67_.clear();
  cl3d_ntc90_.clear();
  cl3d_bdteg_.clear();
  cl3d_quality_.clear();
  cl3d_ipt_.clear();
  cl3d_ienergy_.clear();
}

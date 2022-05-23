#include "DataFormats/L1THGCal/interface/HGCalTower.h"
#include "L1Trigger/L1THGCalUtilities/interface/HGCalTriggerNtupleBase.h"

class HGCalTriggerNtupleHGCTowers : public HGCalTriggerNtupleBase {
public:
  HGCalTriggerNtupleHGCTowers(const edm::ParameterSet& conf);
  ~HGCalTriggerNtupleHGCTowers() override{};
  void initialize(TTree&, const edm::ParameterSet&, edm::ConsumesCollector&&) final;
  void fill(const edm::Event& e, const HGCalTriggerNtupleEventSetup& es) final;

private:
  void clear() final;

  edm::EDGetToken towers_token_;

  int tower_n_;
  std::vector<float> tower_pt_;
  std::vector<float> tower_energy_;
  std::vector<float> tower_eta_;
  std::vector<float> tower_phi_;
  std::vector<float> tower_etEm_;
  std::vector<float> tower_etHad_;
  std::vector<int> tower_iEta_;
  std::vector<int> tower_iPhi_;
};

DEFINE_EDM_PLUGIN(HGCalTriggerNtupleFactory, HGCalTriggerNtupleHGCTowers, "HGCalTriggerNtupleHGCTowers");

HGCalTriggerNtupleHGCTowers::HGCalTriggerNtupleHGCTowers(const edm::ParameterSet& conf) : HGCalTriggerNtupleBase(conf) {
  accessEventSetup_ = false;
}

void HGCalTriggerNtupleHGCTowers::initialize(TTree& tree,
                                             const edm::ParameterSet& conf,
                                             edm::ConsumesCollector&& collector) {
  towers_token_ = collector.consumes<l1t::HGCalTowerBxCollection>(conf.getParameter<edm::InputTag>("Towers"));

  std::string prefix(conf.getUntrackedParameter<std::string>("Prefix", "tower"));

  std::string bname;
  auto withPrefix([&prefix, &bname](char const* vname) -> char const* {
    bname = prefix + "_" + vname;
    return bname.c_str();
  });

  tree.Branch(withPrefix("n"), &tower_n_, (prefix + "_n/I").c_str());
  tree.Branch(withPrefix("pt"), &tower_pt_);
  tree.Branch(withPrefix("energy"), &tower_energy_);
  tree.Branch(withPrefix("eta"), &tower_eta_);
  tree.Branch(withPrefix("phi"), &tower_phi_);
  tree.Branch(withPrefix("etEm"), &tower_etEm_);
  tree.Branch(withPrefix("etHad"), &tower_etHad_);
  tree.Branch(withPrefix("iEta"), &tower_iEta_);
  tree.Branch(withPrefix("iPhi"), &tower_iPhi_);
}

void HGCalTriggerNtupleHGCTowers::fill(const edm::Event& e, const HGCalTriggerNtupleEventSetup& es) {
  // retrieve towers
  edm::Handle<l1t::HGCalTowerBxCollection> towers_h;
  e.getByToken(towers_token_, towers_h);
  const l1t::HGCalTowerBxCollection& towers = *towers_h;

  clear();
  for (auto tower_itr = towers.begin(0); tower_itr != towers.end(0); tower_itr++) {
    tower_n_++;
    // physical values
    tower_pt_.emplace_back(tower_itr->pt());
    tower_energy_.emplace_back(tower_itr->energy());
    tower_eta_.emplace_back(tower_itr->eta());
    tower_phi_.emplace_back(tower_itr->phi());
    tower_etEm_.emplace_back(tower_itr->etEm());
    tower_etHad_.emplace_back(tower_itr->etHad());

    tower_iEta_.emplace_back(tower_itr->id().iEta());
    tower_iPhi_.emplace_back(tower_itr->id().iPhi());
  }
}

void HGCalTriggerNtupleHGCTowers::clear() {
  tower_n_ = 0;
  tower_pt_.clear();
  tower_energy_.clear();
  tower_eta_.clear();
  tower_phi_.clear();
  tower_etEm_.clear();
  tower_etHad_.clear();
  tower_iEta_.clear();
  tower_iPhi_.clear();
}

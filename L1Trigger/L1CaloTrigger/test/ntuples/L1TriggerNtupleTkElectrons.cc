#include "L1TCaloTriggerNtupleBase.h"

#include "DataFormats/L1TCorrelator/interface/TkElectron.h"
#include "DataFormats/L1TCorrelator/interface/TkElectronFwd.h"

class L1TriggerNtupleTkElectrons : public L1TCaloTriggerNtupleBase {
public:
  L1TriggerNtupleTkElectrons(const edm::ParameterSet& conf);
  ~L1TriggerNtupleTkElectrons() override{};
  void initialize(TTree&, const edm::ParameterSet&, edm::ConsumesCollector&&) final;
  void fill(const edm::Event& e, const edm::EventSetup& es) final;

private:
  void clear() final;

  edm::EDGetToken tkEle_token_;

  int tkEle_n_;
  std::vector<float> tkEle_pt_;
  std::vector<float> tkEle_energy_;
  std::vector<float> tkEle_eta_;
  std::vector<float> tkEle_phi_;
  std::vector<float> tkEle_hwQual_;
  std::vector<float> tkEle_tkIso_;
};

DEFINE_EDM_PLUGIN(HGCalTriggerNtupleFactory, L1TriggerNtupleTkElectrons, "L1TriggerNtupleTkElectrons");

L1TriggerNtupleTkElectrons::L1TriggerNtupleTkElectrons(const edm::ParameterSet& conf)
    : L1TCaloTriggerNtupleBase(conf) {}

void L1TriggerNtupleTkElectrons::initialize(TTree& tree,
                                            const edm::ParameterSet& conf,
                                            edm::ConsumesCollector&& collector) {
  tkEle_token_ = collector.consumes<l1t::TkElectronCollection>(conf.getParameter<edm::InputTag>("TkElectrons"));

  tree.Branch(branch_name_w_prefix("n").c_str(), &tkEle_n_, branch_name_w_prefix("n/I").c_str());
  tree.Branch(branch_name_w_prefix("pt").c_str(), &tkEle_pt_);
  tree.Branch(branch_name_w_prefix("energy").c_str(), &tkEle_energy_);
  tree.Branch(branch_name_w_prefix("eta").c_str(), &tkEle_eta_);
  tree.Branch(branch_name_w_prefix("phi").c_str(), &tkEle_phi_);
  tree.Branch(branch_name_w_prefix("hwQual").c_str(), &tkEle_hwQual_);
  tree.Branch(branch_name_w_prefix("tkIso").c_str(), &tkEle_tkIso_);
}

void L1TriggerNtupleTkElectrons::fill(const edm::Event& e, const edm::EventSetup& es) {
  // retrieve towers
  edm::Handle<l1t::TkElectronCollection> tkEle_h;
  e.getByToken(tkEle_token_, tkEle_h);
  const l1t::TkElectronCollection& tkEle_collection = *tkEle_h;

  clear();
  for (auto tkele_itr : tkEle_collection) {
    tkEle_n_++;
    tkEle_pt_.emplace_back(tkele_itr.pt());
    tkEle_energy_.emplace_back(tkele_itr.energy());
    tkEle_eta_.emplace_back(tkele_itr.eta());
    tkEle_phi_.emplace_back(tkele_itr.phi());
    tkEle_hwQual_.emplace_back(tkele_itr.EGRef()->hwQual());
    tkEle_tkIso_.emplace_back(tkele_itr.trkIsol());
  }
}

void L1TriggerNtupleTkElectrons::clear() {
  tkEle_n_ = 0;
  tkEle_pt_.clear();
  tkEle_energy_.clear();
  tkEle_eta_.clear();
  tkEle_phi_.clear();
  tkEle_hwQual_.clear();
  tkEle_tkIso_.clear();
}

#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "L1Trigger/L1THGCalUtilities/interface/HGCalTriggerNtupleBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"

class HGCalTriggerNtupleHGCDigis : public HGCalTriggerNtupleBase {
public:
  HGCalTriggerNtupleHGCDigis(const edm::ParameterSet& conf);
  ~HGCalTriggerNtupleHGCDigis() override{};
  void initialize(TTree&, const edm::ParameterSet&, edm::ConsumesCollector&&) final;
  void fill(const edm::Event& e, const HGCalTriggerNtupleEventSetup& es) final;

private:
  void simhits(const edm::Event& e,
               std::unordered_map<uint32_t, double>& simhits_ee,
               std::unordered_map<uint32_t, double>& simhits_fh,
               std::unordered_map<uint32_t, double>& simhits_bh);
  void clear() final;

  edm::EDGetToken ee_token_, fh_token_, bh_token_;
  bool is_Simhit_comp_;
  edm::EDGetToken SimHits_inputee_, SimHits_inputfh_, SimHits_inputbh_;

  std::vector<unsigned int> digiBXselect_;
  static constexpr unsigned kDigiSize_ = 5;

  HGCalTriggerTools triggerTools_;

  int hgcdigi_n_;
  std::vector<int> hgcdigi_id_;
  std::vector<int> hgcdigi_subdet_;
  std::vector<int> hgcdigi_side_;
  std::vector<int> hgcdigi_layer_;
  std::vector<int> hgcdigi_wafertype_;
  std::vector<float> hgcdigi_eta_;
  std::vector<float> hgcdigi_phi_;
  std::vector<float> hgcdigi_z_;
  std::vector<std::vector<uint32_t>> hgcdigi_data_;
  std::vector<std::vector<int>> hgcdigi_isadc_;
  std::vector<float> hgcdigi_simenergy_;
  std::vector<int> hgcdigi_waferu_;
  std::vector<int> hgcdigi_waferv_;
  std::vector<int> hgcdigi_cellu_;
  std::vector<int> hgcdigi_cellv_;

  int bhdigi_n_;
  std::vector<int> bhdigi_id_;
  std::vector<int> bhdigi_subdet_;
  std::vector<int> bhdigi_side_;
  std::vector<int> bhdigi_layer_;
  std::vector<int> bhdigi_ieta_;
  std::vector<int> bhdigi_iphi_;
  std::vector<float> bhdigi_eta_;
  std::vector<float> bhdigi_phi_;
  std::vector<float> bhdigi_z_;
  std::vector<std::vector<uint32_t>> bhdigi_data_;
  std::vector<std::vector<int>> bhdigi_isadc_;
  std::vector<float> bhdigi_simenergy_;
};

DEFINE_EDM_PLUGIN(HGCalTriggerNtupleFactory, HGCalTriggerNtupleHGCDigis, "HGCalTriggerNtupleHGCDigis");

HGCalTriggerNtupleHGCDigis::HGCalTriggerNtupleHGCDigis(const edm::ParameterSet& conf) : HGCalTriggerNtupleBase(conf) {
  accessEventSetup_ = false;
  is_Simhit_comp_ = conf.getParameter<bool>("isSimhitComp");
  digiBXselect_ = conf.getParameter<std::vector<unsigned int>>("digiBXselect");

  if (digiBXselect_.empty()) {
    throw cms::Exception("BadInitialization") << "digiBXselect vector is empty";
  }
  if (*std::max_element(digiBXselect_.begin(), digiBXselect_.end()) >= kDigiSize_) {
    throw cms::Exception("BadInitialization")
        << "digiBXselect vector requests a BX outside of maximum size of digis (" << kDigiSize_ << " BX)";
  }
  //sort and check for duplicates
  std::sort(digiBXselect_.begin(), digiBXselect_.end());
  if (std::unique(digiBXselect_.begin(), digiBXselect_.end()) != digiBXselect_.end()) {
    throw cms::Exception("BadInitialization") << "digiBXselect vector contains duplicate BX values";
  }
}

void HGCalTriggerNtupleHGCDigis::initialize(TTree& tree,
                                            const edm::ParameterSet& conf,
                                            edm::ConsumesCollector&& collector) {
  ee_token_ = collector.consumes<HGCalDigiCollection>(conf.getParameter<edm::InputTag>("HGCDigisEE"));
  fh_token_ = collector.consumes<HGCalDigiCollection>(conf.getParameter<edm::InputTag>("HGCDigisFH"));
  bh_token_ = collector.consumes<HGCalDigiCollection>(conf.getParameter<edm::InputTag>("HGCDigisBH"));
  if (is_Simhit_comp_) {
    SimHits_inputee_ = collector.consumes<edm::PCaloHitContainer>(conf.getParameter<edm::InputTag>("eeSimHits"));
    SimHits_inputfh_ = collector.consumes<edm::PCaloHitContainer>(conf.getParameter<edm::InputTag>("fhSimHits"));
    SimHits_inputbh_ = collector.consumes<edm::PCaloHitContainer>(conf.getParameter<edm::InputTag>("bhSimHits"));
  }

  hgcdigi_data_.resize(digiBXselect_.size());
  hgcdigi_isadc_.resize(digiBXselect_.size());
  bhdigi_data_.resize(digiBXselect_.size());
  bhdigi_isadc_.resize(digiBXselect_.size());

  tree.Branch("hgcdigi_n", &hgcdigi_n_, "hgcdigi_n/I");
  tree.Branch("hgcdigi_id", &hgcdigi_id_);
  tree.Branch("hgcdigi_subdet", &hgcdigi_subdet_);
  tree.Branch("hgcdigi_zside", &hgcdigi_side_);
  tree.Branch("hgcdigi_layer", &hgcdigi_layer_);
  tree.Branch("hgcdigi_wafertype", &hgcdigi_wafertype_);
  tree.Branch("hgcdigi_eta", &hgcdigi_eta_);
  tree.Branch("hgcdigi_phi", &hgcdigi_phi_);
  tree.Branch("hgcdigi_z", &hgcdigi_z_);
  std::string bname;
  auto withBX([&bname](char const* vname, unsigned int bx) -> char const* {
    bname = std::string(vname) + "_BX" + to_string(bx);
    return bname.c_str();
  });
  for (unsigned int i = 0; i < digiBXselect_.size(); i++) {
    unsigned int bxi = digiBXselect_[i];
    tree.Branch(withBX("hgcdigi_data", bxi), &hgcdigi_data_[i]);
    tree.Branch(withBX("hgcdigi_isadc", bxi), &hgcdigi_isadc_[i]);
  }
  tree.Branch("hgcdigi_waferu", &hgcdigi_waferu_);
  tree.Branch("hgcdigi_waferv", &hgcdigi_waferv_);
  tree.Branch("hgcdigi_cellu", &hgcdigi_cellu_);
  tree.Branch("hgcdigi_cellv", &hgcdigi_cellv_);
  if (is_Simhit_comp_)
    tree.Branch("hgcdigi_simenergy", &hgcdigi_simenergy_);

  tree.Branch("bhdigi_n", &bhdigi_n_, "bhdigi_n/I");
  tree.Branch("bhdigi_id", &bhdigi_id_);
  tree.Branch("bhdigi_subdet", &bhdigi_subdet_);
  tree.Branch("bhdigi_zside", &bhdigi_side_);
  tree.Branch("bhdigi_layer", &bhdigi_layer_);
  tree.Branch("bhdigi_ieta", &bhdigi_ieta_);
  tree.Branch("bhdigi_iphi", &bhdigi_iphi_);
  tree.Branch("bhdigi_eta", &bhdigi_eta_);
  tree.Branch("bhdigi_phi", &bhdigi_phi_);
  tree.Branch("bhdigi_z", &bhdigi_z_);
  for (unsigned int i = 0; i < digiBXselect_.size(); i++) {
    unsigned int bxi = digiBXselect_[i];
    tree.Branch(withBX("bhdigi_data", bxi), &bhdigi_data_[i]);
    tree.Branch(withBX("bhdigi_isadc", bxi), &bhdigi_isadc_[i]);
  }
  if (is_Simhit_comp_)
    tree.Branch("bhdigi_simenergy", &bhdigi_simenergy_);
}

void HGCalTriggerNtupleHGCDigis::fill(const edm::Event& e, const HGCalTriggerNtupleEventSetup& es) {
  edm::Handle<HGCalDigiCollection> ee_digis_h;
  e.getByToken(ee_token_, ee_digis_h);
  const HGCalDigiCollection& ee_digis = *ee_digis_h;
  edm::Handle<HGCalDigiCollection> fh_digis_h;
  e.getByToken(fh_token_, fh_digis_h);
  const HGCalDigiCollection& fh_digis = *fh_digis_h;
  edm::Handle<HGCalDigiCollection> bh_digis_h;
  e.getByToken(bh_token_, bh_digis_h);
  const HGCalDigiCollection& bh_digis = *bh_digis_h;

  triggerTools_.setGeometry(es.geometry.product());

  // sim hit association
  std::unordered_map<uint32_t, double> simhits_ee;
  std::unordered_map<uint32_t, double> simhits_fh;
  std::unordered_map<uint32_t, double> simhits_bh;
  if (is_Simhit_comp_)
    simhits(e, simhits_ee, simhits_fh, simhits_bh);

  clear();
  hgcdigi_n_ = ee_digis.size() + fh_digis.size();
  hgcdigi_id_.reserve(hgcdigi_n_);
  hgcdigi_subdet_.reserve(hgcdigi_n_);
  hgcdigi_side_.reserve(hgcdigi_n_);
  hgcdigi_layer_.reserve(hgcdigi_n_);
  hgcdigi_wafertype_.reserve(hgcdigi_n_);
  hgcdigi_eta_.reserve(hgcdigi_n_);
  hgcdigi_phi_.reserve(hgcdigi_n_);
  hgcdigi_z_.reserve(hgcdigi_n_);
  for (unsigned int i = 0; i < digiBXselect_.size(); i++) {
    hgcdigi_data_[i].reserve(hgcdigi_n_);
    hgcdigi_isadc_[i].reserve(hgcdigi_n_);
  }
  hgcdigi_waferu_.reserve(hgcdigi_n_);
  hgcdigi_waferv_.reserve(hgcdigi_n_);
  hgcdigi_cellu_.reserve(hgcdigi_n_);
  hgcdigi_cellv_.reserve(hgcdigi_n_);
  if (is_Simhit_comp_)
    hgcdigi_simenergy_.reserve(hgcdigi_n_);

  bhdigi_n_ = bh_digis.size();
  bhdigi_id_.reserve(bhdigi_n_);
  bhdigi_subdet_.reserve(bhdigi_n_);
  bhdigi_side_.reserve(bhdigi_n_);
  bhdigi_layer_.reserve(bhdigi_n_);
  bhdigi_ieta_.reserve(bhdigi_n_);
  bhdigi_iphi_.reserve(bhdigi_n_);
  bhdigi_eta_.reserve(bhdigi_n_);
  bhdigi_phi_.reserve(bhdigi_n_);
  bhdigi_z_.reserve(bhdigi_n_);
  for (unsigned int i = 0; i < digiBXselect_.size(); i++) {
    bhdigi_data_[i].reserve(bhdigi_n_);
    bhdigi_isadc_[i].reserve(bhdigi_n_);
  }
  if (is_Simhit_comp_)
    bhdigi_simenergy_.reserve(bhdigi_n_);

  for (const auto& digi : ee_digis) {
    const DetId id(digi.id());
    hgcdigi_id_.emplace_back(id.rawId());
    hgcdigi_subdet_.emplace_back(id.det());
    hgcdigi_side_.emplace_back(triggerTools_.zside(id));
    hgcdigi_layer_.emplace_back(triggerTools_.layerWithOffset(id));
    GlobalPoint cellpos = triggerTools_.getTriggerGeometry()->eeGeometry()->getPosition(id.rawId());
    hgcdigi_eta_.emplace_back(cellpos.eta());
    hgcdigi_phi_.emplace_back(cellpos.phi());
    hgcdigi_z_.emplace_back(cellpos.z());
    for (unsigned int i = 0; i < digiBXselect_.size(); i++) {
      hgcdigi_data_[i].emplace_back(digi[digiBXselect_[i]].data());
      hgcdigi_isadc_[i].emplace_back(!digi[digiBXselect_[i]].mode());
    }
    const HGCSiliconDetId idsi(digi.id());
    hgcdigi_waferu_.emplace_back(idsi.waferU());
    hgcdigi_waferv_.emplace_back(idsi.waferV());
    hgcdigi_wafertype_.emplace_back(idsi.type());
    hgcdigi_cellu_.emplace_back(idsi.cellU());
    hgcdigi_cellv_.emplace_back(idsi.cellV());
    if (is_Simhit_comp_) {
      double hit_energy = 0;
      auto itr = simhits_ee.find(id);
      if (itr != simhits_ee.end())
        hit_energy = itr->second;
      hgcdigi_simenergy_.emplace_back(hit_energy);
    }
  }

  for (const auto& digi : fh_digis) {
    const DetId id(digi.id());
    hgcdigi_id_.emplace_back(id.rawId());
    hgcdigi_subdet_.emplace_back(id.det());
    hgcdigi_side_.emplace_back(triggerTools_.zside(id));
    hgcdigi_layer_.emplace_back(triggerTools_.layerWithOffset(id));
    GlobalPoint cellpos = triggerTools_.getTriggerGeometry()->hsiGeometry()->getPosition(id.rawId());
    hgcdigi_eta_.emplace_back(cellpos.eta());
    hgcdigi_phi_.emplace_back(cellpos.phi());
    hgcdigi_z_.emplace_back(cellpos.z());
    for (unsigned int i = 0; i < digiBXselect_.size(); i++) {
      hgcdigi_data_[i].emplace_back(digi[digiBXselect_[i]].data());
      hgcdigi_isadc_[i].emplace_back(!digi[digiBXselect_[i]].mode());
    }
    const HGCSiliconDetId idsi(digi.id());
    hgcdigi_waferu_.emplace_back(idsi.waferU());
    hgcdigi_waferv_.emplace_back(idsi.waferV());
    hgcdigi_wafertype_.emplace_back(idsi.type());
    hgcdigi_cellu_.emplace_back(idsi.cellU());
    hgcdigi_cellv_.emplace_back(idsi.cellV());
    if (is_Simhit_comp_) {
      double hit_energy = 0;
      auto itr = simhits_fh.find(id);
      if (itr != simhits_fh.end())
        hit_energy = itr->second;
      hgcdigi_simenergy_.emplace_back(hit_energy);
    }
  }

  for (const auto& digi : bh_digis) {
    const DetId id(digi.id());
    bhdigi_id_.emplace_back(id.rawId());
    bhdigi_subdet_.emplace_back(id.det());
    bhdigi_side_.emplace_back(triggerTools_.zside(id));
    bhdigi_layer_.emplace_back(triggerTools_.layerWithOffset(id));
    GlobalPoint cellpos = triggerTools_.getTriggerGeometry()->hscGeometry()->getPosition(id.rawId());
    bhdigi_eta_.emplace_back(cellpos.eta());
    bhdigi_phi_.emplace_back(cellpos.phi());
    bhdigi_z_.emplace_back(cellpos.z());
    for (unsigned int i = 0; i < digiBXselect_.size(); i++) {
      bhdigi_data_[i].emplace_back(digi[digiBXselect_[i]].data());
      bhdigi_isadc_[i].emplace_back(!digi[digiBXselect_[i]].mode());
    }
    const HGCScintillatorDetId idsci(digi.id());
    bhdigi_ieta_.emplace_back(idsci.ietaAbs());
    bhdigi_iphi_.emplace_back(idsci.iphi());
    if (is_Simhit_comp_) {
      double hit_energy = 0;
      auto itr = simhits_bh.find(id);
      if (itr != simhits_bh.end())
        hit_energy = itr->second;
      bhdigi_simenergy_.emplace_back(hit_energy);
    }
  }
}

void HGCalTriggerNtupleHGCDigis::simhits(const edm::Event& e,
                                         std::unordered_map<uint32_t, double>& simhits_ee,
                                         std::unordered_map<uint32_t, double>& simhits_fh,
                                         std::unordered_map<uint32_t, double>& simhits_bh) {
  edm::Handle<edm::PCaloHitContainer> ee_simhits_h;
  e.getByToken(SimHits_inputee_, ee_simhits_h);
  const edm::PCaloHitContainer& ee_simhits = *ee_simhits_h;
  edm::Handle<edm::PCaloHitContainer> fh_simhits_h;
  e.getByToken(SimHits_inputfh_, fh_simhits_h);
  const edm::PCaloHitContainer& fh_simhits = *fh_simhits_h;
  edm::Handle<edm::PCaloHitContainer> bh_simhits_h;
  e.getByToken(SimHits_inputbh_, bh_simhits_h);
  const edm::PCaloHitContainer& bh_simhits = *bh_simhits_h;

  //EE
  for (const auto& simhit : ee_simhits) {
    DetId id = triggerTools_.simToReco(simhit.id(), triggerTools_.getTriggerGeometry()->eeTopology());
    if (id.rawId() == 0)
      continue;
    auto itr_insert = simhits_ee.emplace(id, 0.);
    itr_insert.first->second += simhit.energy();
  }
  //  FH
  for (const auto& simhit : fh_simhits) {
    DetId id = triggerTools_.simToReco(simhit.id(), triggerTools_.getTriggerGeometry()->fhTopology());
    if (id.rawId() == 0)
      continue;
    auto itr_insert = simhits_fh.emplace(id, 0.);
    itr_insert.first->second += simhit.energy();
  }
  //  BH
  for (const auto& simhit : bh_simhits) {
    DetId id = triggerTools_.simToReco(simhit.id(), triggerTools_.getTriggerGeometry()->hscTopology());
    if (id.rawId() == 0)
      continue;
    auto itr_insert = simhits_bh.emplace(id, 0.);
    itr_insert.first->second += simhit.energy();
  }
}

void HGCalTriggerNtupleHGCDigis::clear() {
  hgcdigi_n_ = 0;
  hgcdigi_id_.clear();
  hgcdigi_subdet_.clear();
  hgcdigi_side_.clear();
  hgcdigi_layer_.clear();
  hgcdigi_waferu_.clear();
  hgcdigi_waferv_.clear();
  hgcdigi_wafertype_.clear();
  hgcdigi_cellu_.clear();
  hgcdigi_cellv_.clear();
  hgcdigi_eta_.clear();
  hgcdigi_phi_.clear();
  hgcdigi_z_.clear();
  for (unsigned int i = 0; i < digiBXselect_.size(); i++) {
    hgcdigi_data_[i].clear();
    hgcdigi_isadc_[i].clear();
  }
  if (is_Simhit_comp_)
    hgcdigi_simenergy_.clear();

  bhdigi_n_ = 0;
  bhdigi_id_.clear();
  bhdigi_subdet_.clear();
  bhdigi_side_.clear();
  bhdigi_layer_.clear();
  bhdigi_ieta_.clear();
  bhdigi_iphi_.clear();
  bhdigi_eta_.clear();
  bhdigi_phi_.clear();
  bhdigi_z_.clear();
  for (unsigned int i = 0; i < digiBXselect_.size(); i++) {
    bhdigi_data_[i].clear();
    bhdigi_isadc_[i].clear();
  }
  if (is_Simhit_comp_)
    bhdigi_simenergy_.clear();
}

#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
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
  void fill(const edm::Event& e, const edm::EventSetup& es) final;

private:
  void simhits(const edm::Event& e,
               std::unordered_map<uint32_t, double>& simhits_ee,
               std::unordered_map<uint32_t, double>& simhits_fh,
               std::unordered_map<uint32_t, double>& simhits_bh);
  void clear() final;

  edm::EDGetToken ee_token_, fh_token_, bh_token_;
  bool is_Simhit_comp_;
  edm::EDGetToken SimHits_inputee_, SimHits_inputfh_, SimHits_inputbh_;

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
  std::vector<uint32_t> hgcdigi_data_;
  std::vector<int> hgcdigi_isadc_;
  std::vector<float> hgcdigi_simenergy_;
  // V8 detid scheme
  std::vector<int> hgcdigi_wafer_;
  std::vector<int> hgcdigi_cell_;
  // V9 detid scheme
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
  std::vector<uint32_t> bhdigi_data_;
  std::vector<float> bhdigi_simenergy_;

  edm::ESHandle<HGCalTriggerGeometryBase> triggerGeometry_;
};

DEFINE_EDM_PLUGIN(HGCalTriggerNtupleFactory, HGCalTriggerNtupleHGCDigis, "HGCalTriggerNtupleHGCDigis");

HGCalTriggerNtupleHGCDigis::HGCalTriggerNtupleHGCDigis(const edm::ParameterSet& conf) : HGCalTriggerNtupleBase(conf) {
  is_Simhit_comp_ = conf.getParameter<bool>("isSimhitComp");
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
  tree.Branch("hgcdigi_n", &hgcdigi_n_, "hgcdigi_n/I");
  tree.Branch("hgcdigi_id", &hgcdigi_id_);
  tree.Branch("hgcdigi_subdet", &hgcdigi_subdet_);
  tree.Branch("hgcdigi_zside", &hgcdigi_side_);
  tree.Branch("hgcdigi_layer", &hgcdigi_layer_);
  tree.Branch("hgcdigi_wafertype", &hgcdigi_wafertype_);
  tree.Branch("hgcdigi_eta", &hgcdigi_eta_);
  tree.Branch("hgcdigi_phi", &hgcdigi_phi_);
  tree.Branch("hgcdigi_z", &hgcdigi_z_);
  tree.Branch("hgcdigi_data", &hgcdigi_data_);
  tree.Branch("hgcdigi_isadc", &hgcdigi_isadc_);
  // V9 detid scheme
  tree.Branch("hgcdigi_waferu", &hgcdigi_waferu_);
  tree.Branch("hgcdigi_waferv", &hgcdigi_waferv_);
  tree.Branch("hgcdigi_cellu", &hgcdigi_cellu_);
  tree.Branch("hgcdigi_cellv", &hgcdigi_cellv_);
  // V8 detid scheme
  tree.Branch("hgcdigi_wafer", &hgcdigi_wafer_);
  tree.Branch("hgcdigi_cell", &hgcdigi_cell_);
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
  tree.Branch("bhdigi_data", &bhdigi_data_);
  if (is_Simhit_comp_)
    tree.Branch("bhdigi_simenergy", &bhdigi_simenergy_);
}

void HGCalTriggerNtupleHGCDigis::fill(const edm::Event& e, const edm::EventSetup& es) {
  es.get<CaloGeometryRecord>().get(triggerGeometry_);

  edm::Handle<HGCalDigiCollection> ee_digis_h;
  e.getByToken(ee_token_, ee_digis_h);
  const HGCalDigiCollection& ee_digis = *ee_digis_h;
  edm::Handle<HGCalDigiCollection> fh_digis_h;
  e.getByToken(fh_token_, fh_digis_h);
  const HGCalDigiCollection& fh_digis = *fh_digis_h;
  edm::Handle<HGCalDigiCollection> bh_digis_h;
  e.getByToken(bh_token_, bh_digis_h);
  const HGCalDigiCollection& bh_digis = *bh_digis_h;

  triggerTools_.eventSetup(es);

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
  hgcdigi_data_.reserve(hgcdigi_n_);
  hgcdigi_isadc_.reserve(hgcdigi_n_);
  if (triggerGeometry_->isV9Geometry()) {
    hgcdigi_waferu_.reserve(hgcdigi_n_);
    hgcdigi_waferv_.reserve(hgcdigi_n_);
    hgcdigi_cellu_.reserve(hgcdigi_n_);
    hgcdigi_cellv_.reserve(hgcdigi_n_);
  } else {
    hgcdigi_wafer_.reserve(hgcdigi_n_);
    hgcdigi_cell_.reserve(hgcdigi_n_);
  }
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
  if (is_Simhit_comp_)
    bhdigi_simenergy_.reserve(bhdigi_n_);

  const int kIntimeSample = 2;
  for (const auto& digi : ee_digis) {
    const DetId id(digi.id());
    hgcdigi_id_.emplace_back(id.rawId());
    hgcdigi_subdet_.emplace_back(id.subdetId());
    hgcdigi_side_.emplace_back(triggerTools_.zside(id));
    hgcdigi_layer_.emplace_back(triggerTools_.layerWithOffset(id));
    GlobalPoint cellpos = triggerGeometry_->eeGeometry()->getPosition(id.rawId());
    hgcdigi_eta_.emplace_back(cellpos.eta());
    hgcdigi_phi_.emplace_back(cellpos.phi());
    hgcdigi_z_.emplace_back(cellpos.z());
    hgcdigi_data_.emplace_back(digi[kIntimeSample].data());
    if (triggerGeometry_->isV9Geometry()) {
      const HGCSiliconDetId idv9(digi.id());
      hgcdigi_waferu_.emplace_back(idv9.waferU());
      hgcdigi_waferv_.emplace_back(idv9.waferV());
      hgcdigi_wafertype_.emplace_back(idv9.type());
      hgcdigi_cellu_.emplace_back(idv9.cellU());
      hgcdigi_cellv_.emplace_back(idv9.cellV());
    } else {
      const HGCalDetId idv8(digi.id());
      hgcdigi_wafer_.emplace_back(idv8.wafer());
      hgcdigi_wafertype_.emplace_back(idv8.waferType());
      hgcdigi_cell_.emplace_back(idv8.cell());
    }
    int is_adc = 0;
    if (!(digi[kIntimeSample].mode()))
      is_adc = 1;
    hgcdigi_isadc_.emplace_back(is_adc);
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
    hgcdigi_subdet_.emplace_back(id.subdetId());
    hgcdigi_side_.emplace_back(triggerTools_.zside(id));
    hgcdigi_layer_.emplace_back(triggerTools_.layerWithOffset(id));
    GlobalPoint cellpos = triggerGeometry_->hsiGeometry()->getPosition(id.rawId());
    hgcdigi_eta_.emplace_back(cellpos.eta());
    hgcdigi_phi_.emplace_back(cellpos.phi());
    hgcdigi_z_.emplace_back(cellpos.z());
    hgcdigi_data_.emplace_back(digi[kIntimeSample].data());
    if (triggerGeometry_->isV9Geometry()) {
      const HGCSiliconDetId idv9(digi.id());
      hgcdigi_waferu_.emplace_back(idv9.waferU());
      hgcdigi_waferv_.emplace_back(idv9.waferV());
      hgcdigi_wafertype_.emplace_back(idv9.type());
      hgcdigi_cellu_.emplace_back(idv9.cellU());
      hgcdigi_cellv_.emplace_back(idv9.cellV());
    } else {
      const HGCalDetId idv8(digi.id());
      hgcdigi_wafer_.emplace_back(idv8.wafer());
      hgcdigi_wafertype_.emplace_back(idv8.waferType());
      hgcdigi_cell_.emplace_back(idv8.cell());
    }
    int is_adc = 0;
    if (!(digi[kIntimeSample].mode()))
      is_adc = 1;
    hgcdigi_isadc_.emplace_back(is_adc);
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
    bhdigi_subdet_.emplace_back(id.subdetId());
    bhdigi_side_.emplace_back(triggerTools_.zside(id));
    bhdigi_layer_.emplace_back(triggerTools_.layerWithOffset(id));
    GlobalPoint cellpos = (triggerGeometry_->isV9Geometry() ? triggerGeometry_->hscGeometry()->getPosition(id.rawId())
                                                            : triggerGeometry_->bhGeometry()->getPosition(id.rawId()));
    bhdigi_eta_.emplace_back(cellpos.eta());
    bhdigi_phi_.emplace_back(cellpos.phi());
    bhdigi_z_.emplace_back(cellpos.z());
    bhdigi_data_.emplace_back(digi[kIntimeSample].data());
    if (triggerGeometry_->isV9Geometry()) {
      const HGCScintillatorDetId idv9(digi.id());
      bhdigi_ieta_.emplace_back(idv9.ietaAbs());
      bhdigi_iphi_.emplace_back(idv9.iphi());
    } else {
      const HcalDetId idv8(digi.id());
      bhdigi_ieta_.emplace_back(idv8.ieta());
      bhdigi_iphi_.emplace_back(idv8.iphi());
    }
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
    DetId id = triggerTools_.simToReco(simhit.id(), triggerGeometry_->eeTopology());
    if (id.rawId() == 0)
      continue;
    auto itr_insert = simhits_ee.emplace(id, 0.);
    itr_insert.first->second += simhit.energy();
  }
  //  FH
  for (const auto& simhit : fh_simhits) {
    DetId id = triggerTools_.simToReco(simhit.id(), triggerGeometry_->fhTopology());
    if (id.rawId() == 0)
      continue;
    auto itr_insert = simhits_fh.emplace(id, 0.);
    itr_insert.first->second += simhit.energy();
  }
  //  BH
  for (const auto& simhit : bh_simhits) {
    DetId id =
        (triggerGeometry_->isV9Geometry() ? triggerTools_.simToReco(simhit.id(), triggerGeometry_->hscTopology())
                                          : triggerTools_.simToReco(simhit.id(), triggerGeometry_->bhTopology()));
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
  hgcdigi_wafer_.clear();
  hgcdigi_waferu_.clear();
  hgcdigi_waferv_.clear();
  hgcdigi_wafertype_.clear();
  hgcdigi_cell_.clear();
  hgcdigi_cellu_.clear();
  hgcdigi_cellv_.clear();
  hgcdigi_eta_.clear();
  hgcdigi_phi_.clear();
  hgcdigi_z_.clear();
  hgcdigi_data_.clear();
  hgcdigi_isadc_.clear();
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
  bhdigi_data_.clear();
  if (is_Simhit_comp_)
    bhdigi_simenergy_.clear();
}

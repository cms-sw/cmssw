
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"
#include "Geometry/HcalCommonData/interface/HcalHitRelabeller.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/OneToMany.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticleFwd.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "DataFormats/ForwardDetId/interface/HGCalTriggerDetId.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCalUtilities/interface/HGCalTriggerNtupleBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"

class HGCalTriggerNtupleHGCTriggerCells : public HGCalTriggerNtupleBase {
public:
  HGCalTriggerNtupleHGCTriggerCells(const edm::ParameterSet& conf);
  ~HGCalTriggerNtupleHGCTriggerCells() override{};
  void initialize(TTree&, const edm::ParameterSet&, edm::ConsumesCollector&&) final;
  void fill(const edm::Event& e, const HGCalTriggerNtupleEventSetup& es) final;

private:
  double calibrate(double, int, unsigned);
  void simhits(const edm::Event& e,
               std::unordered_map<uint32_t, double>& simhits_ee,
               std::unordered_map<uint32_t, double>& simhits_fh,
               std::unordered_map<uint32_t, double>& simhits_bh);
  void clear() final;

  HGCalTriggerTools triggerTools_;

  edm::EDGetToken trigger_cells_token_, multiclusters_token_;
  edm::EDGetToken simhits_ee_token_, simhits_fh_token_, simhits_bh_token_;
  edm::EDGetToken caloparticles_map_token_;
  bool fill_simenergy_;
  bool fill_truthmap_;
  bool filter_cells_in_multiclusters_;
  double keV2fC_;
  std::vector<double> fcPerMip_;
  std::vector<double> layerWeights_;
  std::vector<double> thicknessCorrections_;

  int tc_n_;
  std::vector<uint32_t> tc_id_;
  std::vector<int> tc_subdet_;
  std::vector<int> tc_side_;
  std::vector<int> tc_layer_;
  std::vector<int> tc_waferu_;
  std::vector<int> tc_waferv_;
  std::vector<int> tc_wafertype_;
  std::vector<int> tc_cellu_;
  std::vector<int> tc_cellv_;
  std::vector<uint32_t> tc_data_;
  std::vector<uint32_t> tc_uncompressedCharge_;
  std::vector<uint32_t> tc_compressedCharge_;
  std::vector<float> tc_mipPt_;
  std::vector<float> tc_pt_;
  std::vector<float> tc_energy_;
  std::vector<float> tc_simenergy_;
  std::vector<float> tc_eta_;
  std::vector<float> tc_phi_;
  std::vector<float> tc_x_;
  std::vector<float> tc_y_;
  std::vector<float> tc_z_;
  std::vector<uint32_t> tc_cluster_id_;
  std::vector<uint32_t> tc_multicluster_id_;
  std::vector<float> tc_multicluster_pt_;
  std::vector<int> tc_genparticle_index_;

  typedef edm::AssociationMap<edm::OneToMany<CaloParticleCollection, l1t::HGCalTriggerCellBxCollection>> CaloToCellsMap;
};

DEFINE_EDM_PLUGIN(HGCalTriggerNtupleFactory, HGCalTriggerNtupleHGCTriggerCells, "HGCalTriggerNtupleHGCTriggerCells");

HGCalTriggerNtupleHGCTriggerCells::HGCalTriggerNtupleHGCTriggerCells(const edm::ParameterSet& conf)
    : HGCalTriggerNtupleBase(conf),
      fill_simenergy_(conf.getParameter<bool>("FillSimEnergy")),
      fill_truthmap_(conf.getParameter<bool>("FillTruthMap")),
      filter_cells_in_multiclusters_(conf.getParameter<bool>("FilterCellsInMulticlusters")),
      keV2fC_(conf.getParameter<double>("keV2fC")),
      fcPerMip_(conf.getParameter<std::vector<double>>("fcPerMip")),
      layerWeights_(conf.getParameter<std::vector<double>>("layerWeights")),
      thicknessCorrections_(conf.getParameter<std::vector<double>>("thicknessCorrections")) {
  accessEventSetup_ = false;
}

void HGCalTriggerNtupleHGCTriggerCells::initialize(TTree& tree,
                                                   const edm::ParameterSet& conf,
                                                   edm::ConsumesCollector&& collector) {
  trigger_cells_token_ =
      collector.consumes<l1t::HGCalTriggerCellBxCollection>(conf.getParameter<edm::InputTag>("TriggerCells"));
  multiclusters_token_ =
      collector.consumes<l1t::HGCalMulticlusterBxCollection>(conf.getParameter<edm::InputTag>("Multiclusters"));

  if (fill_simenergy_) {
    simhits_ee_token_ = collector.consumes<edm::PCaloHitContainer>(conf.getParameter<edm::InputTag>("eeSimHits"));
    simhits_fh_token_ = collector.consumes<edm::PCaloHitContainer>(conf.getParameter<edm::InputTag>("fhSimHits"));
    simhits_bh_token_ = collector.consumes<edm::PCaloHitContainer>(conf.getParameter<edm::InputTag>("bhSimHits"));
  }

  if (fill_truthmap_)
    caloparticles_map_token_ =
        collector.consumes<CaloToCellsMap>(conf.getParameter<edm::InputTag>("caloParticlesToCells"));

  std::string prefix(conf.getUntrackedParameter<std::string>("Prefix", "tc"));

  std::string bname;
  auto withPrefix([&prefix, &bname](char const* vname) -> char const* {
    bname = prefix + "_" + vname;
    return bname.c_str();
  });

  tree.Branch(withPrefix("n"), &tc_n_, (prefix + "_n/I").c_str());
  tree.Branch(withPrefix("id"), &tc_id_);
  tree.Branch(withPrefix("subdet"), &tc_subdet_);
  tree.Branch(withPrefix("zside"), &tc_side_);
  tree.Branch(withPrefix("layer"), &tc_layer_);
  tree.Branch(withPrefix("waferu"), &tc_waferu_);
  tree.Branch(withPrefix("waferv"), &tc_waferv_);
  tree.Branch(withPrefix("wafertype"), &tc_wafertype_);
  tree.Branch(withPrefix("cellu"), &tc_cellu_);
  tree.Branch(withPrefix("cellv"), &tc_cellv_);
  tree.Branch(withPrefix("data"), &tc_data_);
  tree.Branch(withPrefix("uncompressedCharge"), &tc_uncompressedCharge_);
  tree.Branch(withPrefix("compressedCharge"), &tc_compressedCharge_);
  tree.Branch(withPrefix("pt"), &tc_pt_);
  tree.Branch(withPrefix("mipPt"), &tc_mipPt_);
  tree.Branch(withPrefix("energy"), &tc_energy_);
  if (fill_simenergy_)
    tree.Branch(withPrefix("simenergy"), &tc_simenergy_);
  tree.Branch(withPrefix("eta"), &tc_eta_);
  tree.Branch(withPrefix("phi"), &tc_phi_);
  tree.Branch(withPrefix("x"), &tc_x_);
  tree.Branch(withPrefix("y"), &tc_y_);
  tree.Branch(withPrefix("z"), &tc_z_);
  tree.Branch(withPrefix("cluster_id"), &tc_cluster_id_);
  tree.Branch(withPrefix("multicluster_id"), &tc_multicluster_id_);
  tree.Branch(withPrefix("multicluster_pt"), &tc_multicluster_pt_);
  if (fill_truthmap_)
    tree.Branch(withPrefix("genparticle_index"), &tc_genparticle_index_);
}

void HGCalTriggerNtupleHGCTriggerCells::fill(const edm::Event& e, const HGCalTriggerNtupleEventSetup& es) {
  // retrieve trigger cells
  edm::Handle<l1t::HGCalTriggerCellBxCollection> trigger_cells_h;
  e.getByToken(trigger_cells_token_, trigger_cells_h);
  const l1t::HGCalTriggerCellBxCollection& trigger_cells = *trigger_cells_h;

  // retrieve clusters
  edm::Handle<l1t::HGCalMulticlusterBxCollection> multiclusters_h;
  e.getByToken(multiclusters_token_, multiclusters_h);
  const l1t::HGCalMulticlusterBxCollection& multiclusters = *multiclusters_h;

  // sim hit association
  std::unordered_map<uint32_t, double> simhits_ee;
  std::unordered_map<uint32_t, double> simhits_fh;
  std::unordered_map<uint32_t, double> simhits_bh;
  if (fill_simenergy_)
    simhits(e, simhits_ee, simhits_fh, simhits_bh);

  edm::Handle<CaloToCellsMap> caloparticles_map_h;
  std::unordered_map<uint32_t, unsigned> cell_to_genparticle;
  if (fill_truthmap_) {
    e.getByToken(caloparticles_map_token_, caloparticles_map_h);
    for (auto& keyval : *caloparticles_map_h) {
      for (auto& tcref : keyval.val)
        cell_to_genparticle.emplace(tcref->detId(), keyval.key->g4Tracks().at(0).genpartIndex() - 1);
    }
  }

  // Associate cells to clusters
  std::unordered_map<uint32_t, uint32_t> cell2cluster;
  std::unordered_map<uint32_t, l1t::HGCalMulticlusterBxCollection::const_iterator> cell2multicluster;
  for (auto mcl_itr = multiclusters.begin(0); mcl_itr != multiclusters.end(0); mcl_itr++) {
    // loop on 2D clusters inside 3D clusters
    for (const auto& cl_ptr : mcl_itr->constituents()) {
      // loop on TC inside 2D clusters
      for (const auto& tc_ptr : cl_ptr.second->constituents()) {
        cell2cluster.emplace(tc_ptr.second->detId(), cl_ptr.second->detId());
        cell2multicluster.emplace(tc_ptr.second->detId(), mcl_itr);
      }
    }
  }

  triggerTools_.setGeometry(es.geometry.product());

  clear();
  for (auto tc_itr = trigger_cells.begin(0); tc_itr != trigger_cells.end(0); tc_itr++) {
    if (tc_itr->hwPt() > 0) {
      auto cl_itr = cell2cluster.find(tc_itr->detId());
      auto mcl_itr = cell2multicluster.find(tc_itr->detId());
      uint32_t cl_id = (cl_itr != cell2cluster.end() ? cl_itr->second : 0);
      uint32_t mcl_id = (mcl_itr != cell2multicluster.end() ? mcl_itr->second->detId() : 0);
      float mcl_pt = (mcl_itr != cell2multicluster.end() ? mcl_itr->second->pt() : 0.);
      // Filter cells not included in a multicluster, if requested
      if (filter_cells_in_multiclusters_ && mcl_id == 0)
        continue;
      tc_n_++;
      // hardware data
      DetId id(tc_itr->detId());
      tc_id_.emplace_back(tc_itr->detId());
      tc_side_.emplace_back(triggerTools_.zside(id));
      tc_layer_.emplace_back(triggerTools_.layerWithOffset(id));
      if (id.det() == DetId::HGCalTrigger) {
        HGCalTriggerDetId idtrg(id);
        tc_subdet_.emplace_back(idtrg.subdet());
        tc_waferu_.emplace_back(idtrg.waferU());
        tc_waferv_.emplace_back(idtrg.waferV());
        tc_wafertype_.emplace_back(idtrg.type());
        tc_cellu_.emplace_back(idtrg.triggerCellU());
        tc_cellv_.emplace_back(idtrg.triggerCellV());
      } else if (id.det() == DetId::HGCalHSc) {
        HGCScintillatorDetId idsci(id);
        tc_subdet_.emplace_back(idsci.subdet());
        tc_waferu_.emplace_back(-999);
        tc_waferv_.emplace_back(-999);
        tc_wafertype_.emplace_back(idsci.type());
        tc_cellu_.emplace_back(idsci.ietaAbs());
        tc_cellv_.emplace_back(idsci.iphi());
      } else {
        throw cms::Exception("InvalidHGCalTriggerDetid")
            << "Found unexpected trigger cell detid to be filled in HGCal Trigger Cell ntuple.";
      }
      tc_data_.emplace_back(tc_itr->hwPt());
      tc_uncompressedCharge_.emplace_back(tc_itr->uncompressedCharge());
      tc_compressedCharge_.emplace_back(tc_itr->compressedCharge());
      tc_mipPt_.emplace_back(tc_itr->mipPt());
      // physical values
      tc_pt_.emplace_back(tc_itr->pt());
      tc_energy_.emplace_back(tc_itr->energy());
      tc_eta_.emplace_back(tc_itr->eta());
      tc_phi_.emplace_back(tc_itr->phi());
      tc_x_.emplace_back(tc_itr->position().x());
      tc_y_.emplace_back(tc_itr->position().y());
      tc_z_.emplace_back(tc_itr->position().z());
      // Links between TC and clusters
      tc_cluster_id_.emplace_back(cl_id);
      tc_multicluster_id_.emplace_back(mcl_id);
      tc_multicluster_pt_.emplace_back(mcl_pt);

      if (fill_simenergy_) {
        double energy = 0;
        unsigned layer = triggerTools_.layerWithOffset(id);
        // search for simhit for all the cells inside the trigger cell
        for (uint32_t c_id : triggerTools_.getTriggerGeometry()->getCellsFromTriggerCell(id)) {
          int thickness = triggerTools_.thicknessIndex(c_id);
          if (triggerTools_.isEm(id)) {
            auto itr = simhits_ee.find(c_id);
            if (itr != simhits_ee.end())
              energy += calibrate(itr->second, thickness, layer);
          } else if (triggerTools_.isSilicon(id)) {
            auto itr = simhits_fh.find(c_id);
            if (itr != simhits_fh.end())
              energy += calibrate(itr->second, thickness, layer);
          } else {
            auto itr = simhits_bh.find(c_id);
            if (itr != simhits_bh.end())
              energy += itr->second;
          }
        }
        tc_simenergy_.emplace_back(energy);
      }
    }

    if (fill_truthmap_) {
      auto itr(cell_to_genparticle.find(tc_itr->detId()));
      if (itr == cell_to_genparticle.end())
        tc_genparticle_index_.push_back(-1);
      else
        tc_genparticle_index_.push_back(itr->second);
    }
  }
}

double HGCalTriggerNtupleHGCTriggerCells::calibrate(double energy, int thickness, unsigned layer) {
  double fcPerMip = fcPerMip_[thickness];
  double thicknessCorrection = thicknessCorrections_[thickness];
  double layerWeight = layerWeights_[layer];
  double TeV2GeV = 1.e3;
  return energy * keV2fC_ / fcPerMip * layerWeight * TeV2GeV / thicknessCorrection;
}

void HGCalTriggerNtupleHGCTriggerCells::simhits(const edm::Event& e,
                                                std::unordered_map<uint32_t, double>& simhits_ee,
                                                std::unordered_map<uint32_t, double>& simhits_fh,
                                                std::unordered_map<uint32_t, double>& simhits_bh) {
  edm::Handle<edm::PCaloHitContainer> ee_simhits_h;
  e.getByToken(simhits_ee_token_, ee_simhits_h);
  const edm::PCaloHitContainer& ee_simhits = *ee_simhits_h;
  edm::Handle<edm::PCaloHitContainer> fh_simhits_h;
  e.getByToken(simhits_fh_token_, fh_simhits_h);
  const edm::PCaloHitContainer& fh_simhits = *fh_simhits_h;
  edm::Handle<edm::PCaloHitContainer> bh_simhits_h;
  e.getByToken(simhits_bh_token_, bh_simhits_h);
  const edm::PCaloHitContainer& bh_simhits = *bh_simhits_h;

  // EE
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

void HGCalTriggerNtupleHGCTriggerCells::clear() {
  tc_n_ = 0;
  tc_id_.clear();
  tc_subdet_.clear();
  tc_side_.clear();
  tc_layer_.clear();
  tc_waferu_.clear();
  tc_waferv_.clear();
  tc_wafertype_.clear();
  tc_cellu_.clear();
  tc_cellv_.clear();
  tc_data_.clear();
  tc_uncompressedCharge_.clear();
  tc_compressedCharge_.clear();
  tc_mipPt_.clear();
  tc_pt_.clear();
  tc_energy_.clear();
  tc_simenergy_.clear();
  tc_eta_.clear();
  tc_phi_.clear();
  tc_x_.clear();
  tc_y_.clear();
  tc_z_.clear();
  tc_cluster_id_.clear();
  tc_multicluster_id_.clear();
  tc_multicluster_pt_.clear();
  tc_genparticle_index_.clear();
}

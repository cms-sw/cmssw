
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"
#include "Geometry/HcalCommonData/interface/HcalHitRelabeller.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCalUtilities/interface/HGCalTriggerNtupleBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"

class HGCalTriggerNtupleHGCTriggerCells : public HGCalTriggerNtupleBase {
public:
  HGCalTriggerNtupleHGCTriggerCells(const edm::ParameterSet& conf);
  ~HGCalTriggerNtupleHGCTriggerCells() override{};
  void initialize(TTree&, const edm::ParameterSet&, edm::ConsumesCollector&&) final;
  void fill(const edm::Event& e, const edm::EventSetup& es) final;

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
  bool fill_simenergy_;
  bool filter_cells_in_multiclusters_;
  double keV2fC_;
  std::vector<double> fcPerMip_;
  std::vector<double> layerWeights_;
  std::vector<double> thicknessCorrections_;
  edm::ESHandle<HGCalTriggerGeometryBase> geometry_;

  int tc_n_;
  std::vector<uint32_t> tc_id_;
  std::vector<int> tc_subdet_;
  std::vector<int> tc_side_;
  std::vector<int> tc_layer_;
  std::vector<int> tc_wafer_;
  std::vector<int> tc_wafertype_;
  std::vector<int> tc_cell_;
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
};

DEFINE_EDM_PLUGIN(HGCalTriggerNtupleFactory, HGCalTriggerNtupleHGCTriggerCells, "HGCalTriggerNtupleHGCTriggerCells");

HGCalTriggerNtupleHGCTriggerCells::HGCalTriggerNtupleHGCTriggerCells(const edm::ParameterSet& conf)
    : HGCalTriggerNtupleBase(conf),
      fill_simenergy_(conf.getParameter<bool>("FillSimEnergy")),
      filter_cells_in_multiclusters_(conf.getParameter<bool>("FilterCellsInMulticlusters")) {
  fill_simenergy_ = conf.getParameter<bool>("FillSimEnergy");
  filter_cells_in_multiclusters_ = conf.getParameter<bool>("FilterCellsInMulticlusters");
  keV2fC_ = conf.getParameter<double>("keV2fC");
  fcPerMip_ = conf.getParameter<std::vector<double>>("fcPerMip");
  layerWeights_ = conf.getParameter<std::vector<double>>("layerWeights");
  thicknessCorrections_ = conf.getParameter<std::vector<double>>("thicknessCorrections");
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

  tree.Branch("tc_n", &tc_n_, "tc_n/I");
  tree.Branch("tc_id", &tc_id_);
  tree.Branch("tc_subdet", &tc_subdet_);
  tree.Branch("tc_zside", &tc_side_);
  tree.Branch("tc_layer", &tc_layer_);
  tree.Branch("tc_wafer", &tc_wafer_);
  tree.Branch("tc_wafertype", &tc_wafertype_);
  tree.Branch("tc_cell", &tc_cell_);
  tree.Branch("tc_data", &tc_data_);
  tree.Branch("tc_uncompressedCharge", &tc_uncompressedCharge_);
  tree.Branch("tc_compressedCharge", &tc_compressedCharge_);
  tree.Branch("tc_pt", &tc_pt_);
  tree.Branch("tc_mipPt", &tc_mipPt_);
  tree.Branch("tc_energy", &tc_energy_);
  if (fill_simenergy_)
    tree.Branch("tc_simenergy", &tc_simenergy_);
  tree.Branch("tc_eta", &tc_eta_);
  tree.Branch("tc_phi", &tc_phi_);
  tree.Branch("tc_x", &tc_x_);
  tree.Branch("tc_y", &tc_y_);
  tree.Branch("tc_z", &tc_z_);
  tree.Branch("tc_cluster_id", &tc_cluster_id_);
  tree.Branch("tc_multicluster_id", &tc_multicluster_id_);
  tree.Branch("tc_multicluster_pt", &tc_multicluster_pt_);
}

void HGCalTriggerNtupleHGCTriggerCells::fill(const edm::Event& e, const edm::EventSetup& es) {
  // retrieve trigger cells
  edm::Handle<l1t::HGCalTriggerCellBxCollection> trigger_cells_h;
  e.getByToken(trigger_cells_token_, trigger_cells_h);
  const l1t::HGCalTriggerCellBxCollection& trigger_cells = *trigger_cells_h;

  // retrieve clusters
  edm::Handle<l1t::HGCalMulticlusterBxCollection> multiclusters_h;
  e.getByToken(multiclusters_token_, multiclusters_h);
  const l1t::HGCalMulticlusterBxCollection& multiclusters = *multiclusters_h;

  // retrieve geometry
  es.get<CaloGeometryRecord>().get(geometry_);

  // sim hit association
  std::unordered_map<uint32_t, double> simhits_ee;
  std::unordered_map<uint32_t, double> simhits_fh;
  std::unordered_map<uint32_t, double> simhits_bh;
  if (fill_simenergy_)
    simhits(e, simhits_ee, simhits_fh, simhits_bh);

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

  triggerTools_.eventSetup(es);

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
      HGCalDetId id(tc_itr->detId());
      tc_id_.emplace_back(tc_itr->detId());
      tc_subdet_.emplace_back(id.subdetId());
      tc_side_.emplace_back(id.zside());
      tc_layer_.emplace_back(triggerTools_.layerWithOffset(id));
      tc_wafer_.emplace_back(id.wafer());
      tc_wafertype_.emplace_back(id.waferType());
      tc_cell_.emplace_back(id.cell());
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
        int subdet = id.subdetId();
        unsigned layer = triggerTools_.layerWithOffset(id);
        // search for simhit for all the cells inside the trigger cell
        for (uint32_t c_id : geometry_->getCellsFromTriggerCell(id)) {
          int thickness = triggerTools_.thicknessIndex(c_id);
          switch (subdet) {
            case ForwardSubdetector::HGCEE: {
              auto itr = simhits_ee.find(c_id);
              if (itr != simhits_ee.end())
                energy += calibrate(itr->second, thickness, layer);
              break;
            }
            case ForwardSubdetector::HGCHEF: {
              auto itr = simhits_fh.find(c_id);
              if (itr != simhits_fh.end())
                energy += calibrate(itr->second, thickness, layer);
              break;
            }
            case ForwardSubdetector::HGCHEB: {
              auto itr = simhits_bh.find(c_id);
              if (itr != simhits_bh.end())
                energy += itr->second;
              break;
            }
            default:
              break;
          }
        }
        tc_simenergy_.emplace_back(energy);
      }
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

  //EE
  for (const auto& simhit : ee_simhits) {
    DetId id = triggerTools_.simToReco(simhit.id(), geometry_->eeTopology());
    if (id.rawId() == 0)
      continue;
    auto itr_insert = simhits_ee.emplace(id, 0.);
    itr_insert.first->second += simhit.energy();
  }
  //  FH
  for (const auto& simhit : fh_simhits) {
    DetId id = triggerTools_.simToReco(simhit.id(), geometry_->fhTopology());
    if (id.rawId() == 0)
      continue;
    auto itr_insert = simhits_fh.emplace(id, 0.);
    itr_insert.first->second += simhit.energy();
  }
  //  BH
  for (const auto& simhit : bh_simhits) {
    DetId id = (geometry_->isV9Geometry() ? triggerTools_.simToReco(simhit.id(), geometry_->hscTopology())
                                          : triggerTools_.simToReco(simhit.id(), geometry_->bhTopology()));
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
  tc_wafer_.clear();
  tc_wafertype_.clear();
  tc_cell_.clear();
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
}

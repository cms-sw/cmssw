
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerSums.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/ForwardDetId/interface/HGCalTriggerDetId.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCalUtilities/interface/HGCalTriggerNtupleBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"

class HGCalTriggerNtupleHGCTriggerSums : public HGCalTriggerNtupleBase {
public:
  HGCalTriggerNtupleHGCTriggerSums(const edm::ParameterSet& conf);
  ~HGCalTriggerNtupleHGCTriggerSums() override{};
  void initialize(TTree&, const edm::ParameterSet&, edm::ConsumesCollector&&) final;
  void fill(const edm::Event& e, const edm::EventSetup& es) final;

private:
  void clear() final;

  HGCalTriggerTools triggerTools_;

  edm::EDGetToken trigger_sums_token_;
  edm::ESHandle<HGCalTriggerGeometryBase> geometry_;

  static constexpr unsigned kPanelOffset_ = 0;
  static constexpr unsigned kPanelMask_ = 0x7F;
  static constexpr unsigned kSectorOffset_ = 7;
  static constexpr unsigned kSectorMask_ = 0x7;

  int ts_n_;
  std::vector<uint32_t> ts_id_;
  std::vector<int> ts_subdet_;
  std::vector<int> ts_side_;
  std::vector<int> ts_layer_;
  std::vector<int> ts_panel_number_;
  std::vector<int> ts_panel_sector_;
  std::vector<int> ts_wafer_;
  std::vector<int> ts_wafertype_;
  std::vector<uint32_t> ts_data_;
  std::vector<float> ts_mipPt_;
  std::vector<float> ts_pt_;
  std::vector<float> ts_energy_;
  std::vector<float> ts_eta_;
  std::vector<float> ts_phi_;
  std::vector<float> ts_x_;
  std::vector<float> ts_y_;
  std::vector<float> ts_z_;
};

DEFINE_EDM_PLUGIN(HGCalTriggerNtupleFactory, HGCalTriggerNtupleHGCTriggerSums, "HGCalTriggerNtupleHGCTriggerSums");

HGCalTriggerNtupleHGCTriggerSums::HGCalTriggerNtupleHGCTriggerSums(const edm::ParameterSet& conf)
    : HGCalTriggerNtupleBase(conf) {}

void HGCalTriggerNtupleHGCTriggerSums::initialize(TTree& tree,
                                                  const edm::ParameterSet& conf,
                                                  edm::ConsumesCollector&& collector) {
  trigger_sums_token_ =
      collector.consumes<l1t::HGCalTriggerSumsBxCollection>(conf.getParameter<edm::InputTag>("TriggerSums"));

  std::string prefix(conf.getUntrackedParameter<std::string>("Prefix", "ts"));

  std::string bname;
  auto withPrefix([&prefix, &bname](char const* vname) -> char const* {
    bname = prefix + "_" + vname;
    return bname.c_str();
  });

  tree.Branch(withPrefix("n"), &ts_n_, (prefix + "_n/I").c_str());
  tree.Branch(withPrefix("id"), &ts_id_);
  tree.Branch(withPrefix("subdet"), &ts_subdet_);
  tree.Branch(withPrefix("zside"), &ts_side_);
  tree.Branch(withPrefix("layer"), &ts_layer_);
  tree.Branch(withPrefix("wafer"), &ts_wafer_);
  tree.Branch(withPrefix("wafertype"), &ts_wafertype_);
  tree.Branch(withPrefix("panel_number"), &ts_panel_number_);
  tree.Branch(withPrefix("panel_sector"), &ts_panel_sector_);
  tree.Branch(withPrefix("data"), &ts_data_);
  tree.Branch(withPrefix("pt"), &ts_pt_);
  tree.Branch(withPrefix("mipPt"), &ts_mipPt_);
  tree.Branch(withPrefix("energy"), &ts_energy_);
  tree.Branch(withPrefix("eta"), &ts_eta_);
  tree.Branch(withPrefix("phi"), &ts_phi_);
  tree.Branch(withPrefix("x"), &ts_x_);
  tree.Branch(withPrefix("y"), &ts_y_);
  tree.Branch(withPrefix("z"), &ts_z_);
}

void HGCalTriggerNtupleHGCTriggerSums::fill(const edm::Event& e, const edm::EventSetup& es) {
  // retrieve trigger cells
  edm::Handle<l1t::HGCalTriggerSumsBxCollection> trigger_sums_h;
  e.getByToken(trigger_sums_token_, trigger_sums_h);
  const l1t::HGCalTriggerSumsBxCollection& trigger_sums = *trigger_sums_h;

  // retrieve geometry
  es.get<CaloGeometryRecord>().get(geometry_);

  triggerTools_.eventSetup(es);

  clear();
  for (auto ts_itr = trigger_sums.begin(0); ts_itr != trigger_sums.end(0); ts_itr++) {
    if (ts_itr->pt() > 0) {
      ts_n_++;
      // hardware data
      DetId panelId(ts_itr->detId());
      int panel_sector = -999;
      int panel_number = -999;
      if (panelId.det() == DetId::Forward) {
        HGCalDetId panelIdHGCal(panelId);
        if (panelId.subdetId() == ForwardSubdetector::HGCHEB) {
          panel_number = panelIdHGCal.wafer();
        } else {
          panel_sector = (panelIdHGCal.wafer() >> kSectorOffset_) & kSectorMask_;
          panel_number = (panelIdHGCal.wafer() >> kPanelOffset_) & kPanelMask_;
        }
      } else if (panelId.det() == DetId::HGCalHSc) {
        HGCScintillatorDetId panelIdSci(panelId);
        panel_sector = panelIdSci.iphi();
        panel_number = panelIdSci.ietaAbs();
      }
      ts_panel_number_.emplace_back(panel_number);
      ts_panel_sector_.emplace_back(panel_sector);
      ts_id_.emplace_back(ts_itr->detId());
      ts_side_.emplace_back(triggerTools_.zside(panelId));
      ts_layer_.emplace_back(triggerTools_.layerWithOffset(panelId));
      // V9 detids
      if (panelId.det() == DetId::HGCalTrigger) {
        HGCalTriggerDetId idv9(panelId);
        ts_subdet_.emplace_back(idv9.subdet());
        ts_wafertype_.emplace_back(idv9.type());
      } else if (panelId.det() == DetId::HGCalHSc) {
        HGCScintillatorDetId idv9(panelId);
        ts_subdet_.emplace_back(idv9.subdet());
        ts_wafertype_.emplace_back(idv9.type());
      }
      // V8 detids
      else {
        HGCalDetId idv8(panelId);
        ts_subdet_.emplace_back(panelId.subdetId());
        ts_wafer_.emplace_back(idv8.wafer());
        ts_wafertype_.emplace_back(idv8.waferType());
      }
      ts_data_.emplace_back(ts_itr->hwPt());
      ts_mipPt_.emplace_back(ts_itr->mipPt());
      // physical values
      ts_pt_.emplace_back(ts_itr->pt());
      ts_energy_.emplace_back(ts_itr->energy());
      ts_eta_.emplace_back(ts_itr->eta());
      ts_phi_.emplace_back(ts_itr->phi());
      ts_x_.emplace_back(ts_itr->position().x());
      ts_y_.emplace_back(ts_itr->position().y());
      ts_z_.emplace_back(ts_itr->position().z());
    }
  }
}

void HGCalTriggerNtupleHGCTriggerSums::clear() {
  ts_n_ = 0;
  ts_id_.clear();
  ts_subdet_.clear();
  ts_side_.clear();
  ts_layer_.clear();
  ts_wafer_.clear();
  ts_wafertype_.clear();
  ts_panel_number_.clear();
  ts_panel_sector_.clear();
  ts_data_.clear();
  ts_mipPt_.clear();
  ts_pt_.clear();
  ts_energy_.clear();
  ts_eta_.clear();
  ts_phi_.clear();
  ts_x_.clear();
  ts_y_.clear();
  ts_z_.clear();
}

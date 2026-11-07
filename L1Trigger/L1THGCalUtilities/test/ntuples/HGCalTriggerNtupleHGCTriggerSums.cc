
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeomRotation.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerSums.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/ForwardDetId/interface/HGCalTriggerDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalTriggerModuleDetId.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCalUtilities/interface/HGCalTriggerNtupleBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"
#include <bitset>

class HGCalTriggerNtupleHGCTriggerSums : public HGCalTriggerNtupleBase {
public:
  HGCalTriggerNtupleHGCTriggerSums(const edm::ParameterSet& conf);
  ~HGCalTriggerNtupleHGCTriggerSums() override {}
  void initialize(TTree&, const edm::ParameterSet&, edm::ConsumesCollector&&) final;
  void fill(const edm::Event& e, const HGCalTriggerNtupleEventSetup& es) final;

private:
  void clear() final;
  HGCalGeomRotation::WaferCentring getWaferCentring(unsigned layer, int subdet) const;

  HGCalTriggerTools triggerTools_;

  edm::EDGetToken trigger_sums_token_;
  unsigned scintillatorModulesPerSector_ = 12;

  int ts_n_;
  std::vector<uint32_t> ts_id_;
  std::vector<int> ts_subdet_;
  std::vector<int> ts_side_;
  std::vector<int> ts_layer_;
  std::vector<int> ts_sector_;
  std::vector<int> ts_waferu_;
  std::vector<int> ts_waferv_;
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
    : HGCalTriggerNtupleBase(conf),
      scintillatorModulesPerSector_(conf.getParameter<unsigned>("ScintillatorModulesPerSector")) {
  accessEventSetup_ = false;
}

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
  tree.Branch(withPrefix("sector"), &ts_sector_);
  tree.Branch(withPrefix("waferu"), &ts_waferu_);
  tree.Branch(withPrefix("waferv"), &ts_waferv_);
  tree.Branch(withPrefix("wafertype"), &ts_wafertype_);
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

void HGCalTriggerNtupleHGCTriggerSums::fill(const edm::Event& e, const HGCalTriggerNtupleEventSetup& es) {
  // retrieve trigger cells
  edm::Handle<l1t::HGCalTriggerSumsBxCollection> trigger_sums_h;
  e.getByToken(trigger_sums_token_, trigger_sums_h);
  const l1t::HGCalTriggerSumsBxCollection& trigger_sums = *trigger_sums_h;

  triggerTools_.setGeometry(es.geometry.product());

  clear();
  HGCalGeomRotation rotation(HGCalGeomRotation::SectorType::Sector120Degrees);
  for (auto ts_itr = trigger_sums.begin(0); ts_itr != trigger_sums.end(0); ts_itr++) {
    if (ts_itr->pt() > 0) {
      ts_n_++;
      // hardware data
      DetId moduleId(ts_itr->detId());
      ts_id_.emplace_back(ts_itr->detId());
      ts_side_.emplace_back(triggerTools_.zside(moduleId));
      ts_layer_.emplace_back(triggerTools_.layerWithOffset(moduleId));
      if (moduleId.subdetId() == ForwardSubdetector::HGCTrigger) {
        HGCalTriggerModuleDetId id(moduleId);
        int subdet(id.triggerSubdetId());
        if (subdet == HGCalTriggerSubdetector::HGCalHScTrigger) {
          int phi = id.phi() + scintillatorModulesPerSector_ * id.sector();
          ts_waferu_.emplace_back(id.eta());
          ts_waferv_.emplace_back(phi);
        } else {  // silicon
          int sector = id.sector();
          int modU = id.moduleU();
          int modV = id.moduleV();
          rotation.uvMappingFromSector0(getWaferCentring(id.layer(), subdet), modU, modV, sector);
          ts_waferu_.emplace_back(modU);
          ts_waferv_.emplace_back(modV);
        }
        ts_subdet_.emplace_back(subdet);
        ts_sector_.emplace_back(id.sector());
        ts_wafertype_.emplace_back(id.type());
      } else {
        ts_subdet_.emplace_back(-999);
        ts_sector_.emplace_back(-999);
        ts_waferu_.emplace_back(-999);
        ts_waferv_.emplace_back(-999);
        ts_wafertype_.emplace_back(-999);
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
  ts_sector_.clear();
  ts_waferu_.clear();
  ts_waferv_.clear();
  ts_wafertype_.clear();
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

HGCalGeomRotation::WaferCentring HGCalTriggerNtupleHGCTriggerSums::getWaferCentring(unsigned layer, int subdet) const {
  if (subdet == HGCalTriggerSubdetector::HGCalEETrigger) {  // CE-E
    return HGCalGeomRotation::WaferCentring::WaferCentred;  // All layers are wafer centred
  } else if (subdet == HGCalTriggerSubdetector::HGCalHSiTrigger) {
    auto topologyLayerType = triggerTools_.getTriggerGeometry()->hsiTopology().dddConstants().layerType(layer);

    if (topologyLayerType == HGCalTypes::CornerCenterYp)
      return HGCalGeomRotation::WaferCentring::CornerCentredMercedes;
    else if (topologyLayerType == HGCalTypes::CornerCenterYm)
      return HGCalGeomRotation::WaferCentring::CornerCentredY;
    else
      return HGCalGeomRotation::WaferCentring::WaferCentred;

  } else if (subdet == HGCalTriggerSubdetector::HFNoseTrigger) {  //HFNose
    return HGCalGeomRotation::WaferCentring::WaferCentred;
  } else {
    edm::LogError("HGCalTriggerNtupleHGCTriggerSums")
        << "HGCalTriggerNtupleHGCTriggerSums: trigger sub-detector expected to be silicon";
    return HGCalGeomRotation::WaferCentring::WaferCentred;
  }
}

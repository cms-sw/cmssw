
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/L1THGCal/interface/HGCalConcentratorData.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/ForwardDetId/interface/HGCalTriggerDetId.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCalUtilities/interface/HGCalTriggerNtupleBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"

class HGCalTriggerNtupleHGCConcentratorData : public HGCalTriggerNtupleBase {
public:
  HGCalTriggerNtupleHGCConcentratorData(const edm::ParameterSet& conf);
  ~HGCalTriggerNtupleHGCConcentratorData() override{};
  void initialize(TTree&, const edm::ParameterSet&, edm::ConsumesCollector&&) final;
  void fill(const edm::Event& e, const HGCalTriggerNtupleEventSetup& es) final;

private:
  void clear() final;

  HGCalTriggerTools triggerTools_;

  edm::EDGetToken concentrator_data_token_;

  int econ_n_;
  std::vector<uint32_t> econ_id_;
  std::vector<int> econ_subdet_;
  std::vector<int> econ_side_;
  std::vector<int> econ_layer_;
  std::vector<int> econ_waferu_;
  std::vector<int> econ_waferv_;
  std::vector<int> econ_wafertype_;
  std::vector<uint32_t> econ_index_;
  std::vector<uint32_t> econ_data_;
};

DEFINE_EDM_PLUGIN(HGCalTriggerNtupleFactory,
                  HGCalTriggerNtupleHGCConcentratorData,
                  "HGCalTriggerNtupleHGCConcentratorData");

HGCalTriggerNtupleHGCConcentratorData::HGCalTriggerNtupleHGCConcentratorData(const edm::ParameterSet& conf)
    : HGCalTriggerNtupleBase(conf) {
  accessEventSetup_ = false;
}

void HGCalTriggerNtupleHGCConcentratorData::initialize(TTree& tree,
                                                       const edm::ParameterSet& conf,
                                                       edm::ConsumesCollector&& collector) {
  concentrator_data_token_ =
      collector.consumes<l1t::HGCalConcentratorDataBxCollection>(conf.getParameter<edm::InputTag>("ConcentratorData"));

  std::string prefix(conf.getUntrackedParameter<std::string>("Prefix", "econ"));

  std::string bname;
  auto withPrefix([&prefix, &bname](char const* vname) -> char const* {
    bname = prefix + "_" + vname;
    return bname.c_str();
  });

  tree.Branch(withPrefix("n"), &econ_n_, (prefix + "_n/I").c_str());
  tree.Branch(withPrefix("id"), &econ_id_);
  tree.Branch(withPrefix("subdet"), &econ_subdet_);
  tree.Branch(withPrefix("zside"), &econ_side_);
  tree.Branch(withPrefix("layer"), &econ_layer_);
  tree.Branch(withPrefix("waferu"), &econ_waferu_);
  tree.Branch(withPrefix("waferv"), &econ_waferv_);
  tree.Branch(withPrefix("wafertype"), &econ_wafertype_);
  tree.Branch(withPrefix("index"), &econ_index_);
  tree.Branch(withPrefix("data"), &econ_data_);
}

void HGCalTriggerNtupleHGCConcentratorData::fill(const edm::Event& e, const HGCalTriggerNtupleEventSetup& es) {
  // retrieve trigger cells
  edm::Handle<l1t::HGCalConcentratorDataBxCollection> concentrator_data_h;
  e.getByToken(concentrator_data_token_, concentrator_data_h);
  const l1t::HGCalConcentratorDataBxCollection& concentrator_data = *concentrator_data_h;

  triggerTools_.setGeometry(es.geometry.product());

  clear();
  for (auto econ_itr = concentrator_data.begin(0); econ_itr != concentrator_data.end(0); econ_itr++) {
    econ_n_++;
    // hardware data
    DetId id(econ_itr->detId());
    econ_id_.emplace_back(econ_itr->detId());
    econ_side_.emplace_back(triggerTools_.zside(id));
    econ_layer_.emplace_back(triggerTools_.layerWithOffset(id));

    HGCalTriggerDetId idv9(id);
    econ_subdet_.emplace_back(idv9.subdet());
    econ_waferu_.emplace_back(idv9.waferU());
    econ_waferv_.emplace_back(idv9.waferV());
    econ_wafertype_.emplace_back(idv9.type());

    econ_index_.emplace_back(econ_itr->index());
    econ_data_.emplace_back(econ_itr->data());
  }
}

void HGCalTriggerNtupleHGCConcentratorData::clear() {
  econ_n_ = 0;
  econ_id_.clear();
  econ_subdet_.clear();
  econ_side_.clear();
  econ_layer_.clear();
  econ_wafertype_.clear();
  econ_data_.clear();
}

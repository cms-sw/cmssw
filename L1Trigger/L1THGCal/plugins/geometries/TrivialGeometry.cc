#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryGenericMapping.h"

class TrivialGeometry : public HGCalTriggerGeometryGenericMapping {
public:
  TrivialGeometry(const edm::ParameterSet& conf) : HGCalTriggerGeometryGenericMapping(conf) {}

  void initialize(const CaloGeometry*) final {
    constexpr unsigned nmodules = 6;
    for (unsigned i = 0; i < nmodules; ++i) {
      trigger_cells_to_modules_[i] = i;

      HGCalTriggerGeometry::TriggerCell::list_type tc_empty;
      trigger_cells_[i] = std::make_unique<HGCalTriggerGeometry::TriggerCell>(i, i, GlobalPoint(), tc_empty, tc_empty);

      HGCalTriggerGeometry::Module::list_type mod_empty;
      HGCalTriggerGeometry::Module::list_type mod_comps = {i};
      HGCalTriggerGeometry::Module::tc_map_type map_empty;
      modules_[i] = std::make_unique<HGCalTriggerGeometry::Module>(i, GlobalPoint(), mod_empty, mod_comps, map_empty);
    }
  }

  void initialize(const HGCalGeometry*, const HGCalGeometry*, const HGCalGeometry*) final {}
  void initialize(const HGCalGeometry*, const HGCalGeometry*, const HGCalGeometry*, const HGCalGeometry*) final {}
};

DEFINE_EDM_PLUGIN(HGCalTriggerGeometryFactory, TrivialGeometry, "TrivialGeometry");

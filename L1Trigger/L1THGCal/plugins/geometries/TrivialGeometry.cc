#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

class TrivialGeometry : public HGCalTriggerGeometryBase {
public:
  TrivialGeometry(const edm::ParameterSet& conf) :
    HGCalTriggerGeometryBase(conf) {
  }

  virtual void initialize(const es_info& ) override final {
    constexpr unsigned nmodules = 6;
    for( unsigned i = 0; i < nmodules; ++i ) {
      trigger_cells_to_modules_[i] = i;

      HGCalTriggerGeometry::TriggerCell::list_type tc_empty;
      trigger_cells_[i].reset( new HGCalTriggerGeometry::TriggerCell(i,i,
                                                                     GlobalPoint(),
                                                                     tc_empty,
                                                                     tc_empty) );
      
      HGCalTriggerGeometry::Module::list_type mod_empty;
      HGCalTriggerGeometry::Module::list_type mod_comps = { i };
      HGCalTriggerGeometry::Module::tc_map_type map_empty;
      modules_[i].reset( new HGCalTriggerGeometry::Module(i,GlobalPoint(),
                                                          mod_empty,
                                                          mod_comps,
                                                          map_empty) );
    }
  }
};

DEFINE_EDM_PLUGIN(HGCalTriggerGeometryFactory, 
                  TrivialGeometry,
                  "TrivialGeometry");

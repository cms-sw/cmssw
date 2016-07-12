#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

class NullGeometry : public HGCalTriggerGeometryBase {
public:
  NullGeometry(const edm::ParameterSet& conf) :
    HGCalTriggerGeometryBase(conf) {
  }

  virtual void initialize(const es_info& ) override final {}
};

DEFINE_EDM_PLUGIN(HGCalTriggerGeometryFactory, 
                  NullGeometry,
                  "NullGeometry");

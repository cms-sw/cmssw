#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryOld.h"

class NullGeometry : public HGCalTriggerGeometryOld {
public:
  NullGeometry(const edm::ParameterSet& conf) :
    HGCalTriggerGeometryOld(conf) {
  }

  virtual void initialize(const es_info& ) override final {}
};

DEFINE_EDM_PLUGIN(HGCalTriggerGeometryFactory, 
                  NullGeometry,
                  "NullGeometry");

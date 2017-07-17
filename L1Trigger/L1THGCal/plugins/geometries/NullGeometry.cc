#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryGenericMapping.h"

class NullGeometry : public HGCalTriggerGeometryGenericMapping {
public:
  NullGeometry(const edm::ParameterSet& conf) :
    HGCalTriggerGeometryGenericMapping(conf) {
  }

  virtual void initialize(const edm::ESHandle<CaloGeometry>& ) override final {}
};

DEFINE_EDM_PLUGIN(HGCalTriggerGeometryFactory, 
                  NullGeometry,
                  "NullGeometry");

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryGenericMapping.h"

class NullGeometry : public HGCalTriggerGeometryGenericMapping {
public:
  NullGeometry(const edm::ParameterSet& conf) :
    HGCalTriggerGeometryGenericMapping(conf) {
  }

  void initialize(const edm::ESHandle<CaloGeometry>& ) final {}
  void initialize(const edm::ESHandle<HGCalGeometry>&,
          const edm::ESHandle<HGCalGeometry>&,
          const edm::ESHandle<HGCalGeometry>&) final {}
};

DEFINE_EDM_PLUGIN(HGCalTriggerGeometryFactory, 
                  NullGeometry,
                  "NullGeometry");

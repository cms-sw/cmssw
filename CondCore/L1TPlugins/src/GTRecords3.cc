#include "CondCore/ESSources/interface/registration_macros.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"

namespace {
  struct L1GtTriggerMenuInitializer {
    void operator()(L1GtTriggerMenu& _) { _.buildGtConditionMap(); }
  };
}  // namespace
REGISTER_PLUGIN_INIT(L1GtTriggerMenuRcd, L1GtTriggerMenu, L1GtTriggerMenuInitializer);

#include "CondFormats/L1TObjects/interface/L1GtPsbSetup.h"
#include "CondFormats/DataRecord/interface/L1GtPsbSetupRcd.h"

REGISTER_PLUGIN(L1GtPsbSetupRcd, L1GtPsbSetup);

#include "CondFormats/L1TObjects/interface/L1CaloGeometry.h"
#include "CondFormats/DataRecord/interface/L1CaloGeometryRecord.h"

REGISTER_PLUGIN(L1CaloGeometryRecord, L1CaloGeometry);

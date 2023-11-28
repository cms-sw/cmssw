/*
 *  plugin.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 8/16/23.
 *
 */

#include "CondFormats/Common/interface/BaseKeyed.h"
#include "CondFormats/Calibration/interface/Conf.h"
#include "CondFormats/DataRecord/interface/ExEfficiency.h"

#include "CondFormats/DTObjects/interface/DTKeyedConfig.h"
#include "CondFormats/DataRecord/interface/DTKeyedConfigContainerRcd.h"
#include "CondFormats/DataRecord/interface/DTKeyedConfigListRcd.h"

#include "CondCore/ESSources/interface/registration_macros.h"
#include "CondCore/CondDB/interface/KeyListProxy.h"

namespace cond {
  template <>
  std::unique_ptr<BaseKeyed> deserialize<BaseKeyed>(const std::string& payloadType,
                                                    const Binary& payloadData,
                                                    const Binary& streamerInfoData) {
    DESERIALIZE_BASE_CASE(BaseKeyed);
    DESERIALIZE_POLIMORPHIC_CASE(BaseKeyed, condex::ConfI);
    DESERIALIZE_POLIMORPHIC_CASE(BaseKeyed, DTKeyedConfig);

    // here we come if none of the deserializations above match the payload type:
    throwException(std::string("Type mismatch, target object is type \"") + payloadType + "\"", "deserialize<>");
  }
}  // namespace cond

namespace cond::serialization {
  template <>
  struct BaseClassInfo<BaseKeyed> {
    constexpr static bool kAbstract = false;
    using inheriting_classes_t = edm::mpl::Vector<condex::ConfI, DTKeyedConfig>;
  };
}  // namespace cond::serialization

DEFINE_COND_CLASSNAME(condex::ConfI)
DEFINE_COND_CLASSNAME(DTKeyedConfig)

DEFINE_COND_SERIAL_REGISTER_PLUGIN(cond::BaseKeyed);

REGISTER_PLUGIN_NO_SERIAL(ExDwarfRcd, cond::BaseKeyed);
REGISTER_KEYLIST_PLUGIN(ExDwarfListRcd, cond::persistency::KeyList, ExDwarfRcd);

REGISTER_PLUGIN_NO_SERIAL(DTKeyedConfigContainerRcd, cond::BaseKeyed);
REGISTER_KEYLIST_PLUGIN(DTKeyedConfigListRcd, cond::persistency::KeyList, DTKeyedConfigContainerRcd);

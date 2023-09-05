/*
 *  plugin.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 7/24/05.
 *
 */

#include "CondCore/ESSources/interface/registration_macros.h"
#include "CondFormats/Calibration/interface/Pedestals.h"
#include "CondFormats/Calibration/interface/Conf.h"
#include "CondFormats/DataRecord/interface/PedestalsRcd.h"
#include "CondFormats/Calibration/interface/mySiStripNoises.h"
#include "CondFormats/DataRecord/interface/mySiStripNoisesRcd.h"
#include "CondFormats/Calibration/interface/Efficiency.h"
#include "CondFormats/DataRecord/interface/ExEfficiency.h"

//
#include "CondCore/CondDB/interface/Serialization.h"

namespace cond {
  template <>
  std::unique_ptr<condex::Efficiency> deserialize<condex::Efficiency>(const std::string& payloadType,
                                                                      const Binary& payloadData,
                                                                      const Binary& streamerInfoData) {
    // DESERIALIZE_BASE_CASE( condex::Efficiency ); abstract
    DESERIALIZE_POLIMORPHIC_CASE(condex::Efficiency, condex::ParametricEfficiencyInPt);
    DESERIALIZE_POLIMORPHIC_CASE(condex::Efficiency, condex::ParametricEfficiencyInEta);

    // here we come if none of the deserializations above match the payload type:
    throwException(std::string("Type mismatch, target object is type \"") + payloadType + "\"", "deserialize<>");
  }
}  // namespace cond

namespace {
  struct InitEfficiency {
    void operator()(condex::Efficiency& e) { e.initialize(); }
  };
}  // namespace

namespace cond::serialization {
  template <>
  struct BaseClassInfo<condex::Efficiency> {
    constexpr static bool kAbstract = true;
    using inheriting_classes_t = edm::mpl::Vector<condex::ParametricEfficiencyInPt, condex::ParametricEfficiencyInEta>;
  };

}  // namespace cond::serialization

DEFINE_COND_CLASSNAME(condex::ParametricEfficiencyInPt)
DEFINE_COND_CLASSNAME(condex::ParametricEfficiencyInEta)

REGISTER_PLUGIN(PedestalsRcd, Pedestals);
REGISTER_PLUGIN_NO_SERIAL(anotherPedestalsRcd, Pedestals);
REGISTER_PLUGIN(mySiStripNoisesRcd, mySiStripNoises);
REGISTER_PLUGIN_INIT(ExEfficiencyRcd, condex::Efficiency, InitEfficiency);

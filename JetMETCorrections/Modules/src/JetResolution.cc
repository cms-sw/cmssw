#ifndef STANDALONE
#include <JetMETCorrections/Modules/interface/JetResolution.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/ESHandle.h>

#include <CondFormats/DataRecord/interface/JetResolutionRcd.h>
#include <CondFormats/DataRecord/interface/JetResolutionScaleFactorRcd.h>
#else
#include "JetResolution.h"
#endif

namespace JME {

  JetResolution::JetResolution(const std::string& filename) {
    m_object = std::make_shared<JetResolutionObject>(filename);
  }

  JetResolution::JetResolution(const JetResolutionObject& object) {
    m_object = std::make_shared<JetResolutionObject>(object);
  }

#ifndef STANDALONE
  const JetResolution JetResolution::get(const edm::EventSetup& setup, const std::string& label) {
    edm::ESHandle<JetResolutionObject> handle;
    setup.get<JetResolutionRcd>().get(label, handle);

    return *handle;
  }
#endif

  float JetResolution::getResolution(const JetParameters& parameters) const {
    const JetResolutionObject::Record* record = m_object->getRecord(parameters);
    if (!record)
      return 1;

    return m_object->evaluateFormula(*record, parameters);
  }

  JetResolutionScaleFactor::JetResolutionScaleFactor(const std::string& filename) {
    m_object = std::make_shared<JetResolutionObject>(filename);
  }

  JetResolutionScaleFactor::JetResolutionScaleFactor(const JetResolutionObject& object) {
    m_object = std::make_shared<JetResolutionObject>(object);
  }

#ifndef STANDALONE
  const JetResolutionScaleFactor JetResolutionScaleFactor::get(const edm::EventSetup& setup, const std::string& label) {
    edm::ESHandle<JetResolutionObject> handle;
    setup.get<JetResolutionScaleFactorRcd>().get(label, handle);

    return *handle;
  }
#endif

  float JetResolutionScaleFactor::getScaleFactor(const JetParameters& parameters,
                                                 Variation variation /* = Variation::NOMINAL*/,
                                                 std::string uncertaintySource /* = ""*/) const {
    const JetResolutionObject::Record* record = m_object->getRecord(parameters);
    if (!record)
      return 1;

    const std::vector<float>& parameters_values = record->getParametersValues();
    const std::vector<std::string>& parameter_names = m_object->getDefinition().getParametersName();
    size_t parameter = static_cast<size_t>(variation);
    if (!uncertaintySource.empty()) {
      if (variation == Variation::DOWN)
        parameter =
            std::distance(parameter_names.begin(),
                          std::find(parameter_names.begin(), parameter_names.end(), uncertaintySource + "Down"));
      else if (variation == Variation::UP)
        parameter = std::distance(parameter_names.begin(),
                                  std::find(parameter_names.begin(), parameter_names.end(), uncertaintySource + "Up"));
      if (parameter >= parameter_names.size()) {
        std::string s;
        for (const auto& piece : parameter_names)
          s += piece + " ";
        throw cms::Exception("InvalidParameter")
            << "Invalid value for 'uncertaintySource' parameter. Only " + s + " are supported.\n";
      }
    }
    return parameters_values[parameter];
  }

}  // namespace JME

#ifndef STANDALONE
#include "FWCore/Utilities/interface/typelookup.h"
TYPELOOKUP_DATA_REG(JME::JetResolution);
TYPELOOKUP_DATA_REG(JME::JetResolutionScaleFactor);
#endif

// CepGen-CMSSW interfacing module
//   2022-2024, Laurent Forthomme

#ifndef GeneratorInterface_CepGenInterface_CepGenParametersConverter_h
#define GeneratorInterface_CepGenInterface_CepGenParametersConverter_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <CepGen/Core/ParametersList.h>

namespace cepgen {
  ParametersList fromParameterSet(const edm::ParameterSet& iConfig) {
    ParametersList params;
    for (const auto& param : iConfig.getParameterNames()) {
      const auto cepgen_param = param == "name" ? MODULE_NAME : param;
      if (iConfig.existsAs<bool>(param))
        params.set(cepgen_param, iConfig.getParameter<bool>(param));
      if (iConfig.existsAs<int>(param))
        params.set(cepgen_param, iConfig.getParameter<int>(param));
      if (iConfig.existsAs<unsigned>(param))
        params.set<unsigned long long>(cepgen_param, iConfig.getParameter<unsigned>(param));
      if (iConfig.existsAs<double>(param))
        params.set(cepgen_param, iConfig.getParameter<double>(param));
      if (iConfig.existsAs<std::string>(param))
        params.set(cepgen_param, iConfig.getParameter<std::string>(param));
      if (iConfig.existsAs<std::vector<double> >(param)) {
        const auto& vec = iConfig.getParameter<std::vector<double> >(param);
        if (vec.size() == 2)
          params.set<Limits>(cepgen_param, Limits{vec.at(0), vec.at(1)});
        params.set(cepgen_param, iConfig.getParameter<std::vector<double> >(param));
      }
      if (iConfig.existsAs<std::vector<std::string> >(param))
        params.set(cepgen_param, iConfig.getParameter<std::vector<std::string> >(param));
      if (iConfig.existsAs<edm::ParameterSet>(param))
        params.set(cepgen_param, fromParameterSet(iConfig.getParameter<edm::ParameterSet>(param)));
    }
    return params;
  }
}  // namespace cepgen

#endif

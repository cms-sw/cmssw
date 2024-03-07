// CepGen-CMSSW interfacing module
//   2022-2024, Laurent Forthomme

#ifndef GeneratorInterface_CepGenInterface_CepGenEventGenerator_h
#define GeneratorInterface_CepGenInterface_CepGenEventGenerator_h

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "GeneratorInterface/Core/interface/BaseHadronizer.h"

#include <CepGen/Generator.h>

namespace gen {
  class CepGenEventGenerator : public BaseHadronizer {
  public:
    explicit CepGenEventGenerator(const edm::ParameterSet&, edm::ConsumesCollector&&);
    virtual ~CepGenEventGenerator();

    bool readSettings(int) { return true; }
    bool declareStableParticles(const std::vector<int>&) { return true; }
    bool declareSpecialSettings(const std::vector<std::string>&) { return true; }

    bool initializeForInternalPartons();
    bool generatePartonsAndHadronize();
    bool decay() { return true; }  // NOT used - let's call it "design imperfection"
    bool residualDecay() { return true; }

    void finalizeEvent() {}
    void statistics() {}

    const char* classname() const { return "CepGenEventGenerator"; }
    const std::vector<std::string>& doSharedResources() const override { return shared_resources_; }

  private:
    cepgen::Generator* gen_{nullptr};
    const cepgen::ParametersList proc_params_;
    std::vector<std::pair<std::string, cepgen::ParametersList> > modif_modules_, output_modules_;
    const std::vector<std::string> shared_resources_;
    HepMC::GenCrossSection xsec_;
  };
}  // namespace gen

#endif

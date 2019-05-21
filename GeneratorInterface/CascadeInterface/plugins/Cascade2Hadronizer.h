#ifndef gen_Cascade2Hadronizer_h
#define gen_Cascade2Hadronizer_h

#include <memory>
#include <string>
#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "GeneratorInterface/Core/interface/ParameterCollector.h"
#include "GeneratorInterface/Core/interface/BaseHadronizer.h"

namespace HepMC {
  class GenEvent;
}

namespace CLHEP {
  class HepRandomEngine;
}

namespace gen {

  class Pythia6Service;

  class Cascade2Hadronizer : public BaseHadronizer {
  public:
    Cascade2Hadronizer(edm::ParameterSet const& ps);
    ~Cascade2Hadronizer() override;

    bool readSettings(int);
    bool initializeForExternalPartons();  //-- initializer for the LHE input
    bool initializeForInternalPartons();

    //-- Read the parameters and pass them to the common blocks
    bool cascadeReadParameters(const std::string& ParameterString);
    void cascadePrintParameters();
    void pythia6PrintParameters();
    bool declareStableParticles(const std::vector<int>&);
    bool declareSpecialSettings(const std::vector<std::string>&) { return true; }
    void statistics();

    bool generatePartonsAndHadronize();
    bool hadronize();  //-- hadronizer for the LHE input
    bool decay();
    bool residualDecay();
    void finalizeEvent();

    const char* classname() const;

  private:
    void doSetRandomEngine(CLHEP::HepRandomEngine* v) override;
    std::vector<std::string> const& doSharedResources() const override { return theSharedResources; }

    static const std::vector<std::string> theSharedResources;

    //-- methods

    void flushTmpStorage();
    void fillTmpStorage();
    void imposeProperTime();  //-- to correctly treat particle decay

    //-- data members

    edm::ParameterSet fParameters;

    Pythia6Service* fPy6Service;

    double
        fComEnergy;  //-- irrelevant for setting py6 as hadronizer (or if anything, it should be picked up from LHERunInfoProduct)
    double fextCrossSection;
    double fextCrossSectionError;
    double fFilterEfficiency;

    unsigned int fMaxEventsToPrint;
    bool fHepMCVerbosity;
    unsigned int fPythiaListVerbosity;  //-- p6 specific

    bool fDisplayPythiaBanner;  //-- p6 specific
    bool fDisplayPythiaCards;   //-- p6 specific

    bool fConvertToPDG;  //-- conversion of Py6 PID's into PDG convention
  };
}  // namespace gen

#endif

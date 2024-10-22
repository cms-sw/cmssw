// -*- C++ -*-
//
//

//
// This class is a "Hadronizer" template (see GeneratorInterface/Core)
//

#ifndef gen_Py8GunBase_h
#define gen_Py8GunBase_h

#include <memory>
#include <string>
#include <vector>

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "GeneratorInterface/Pythia8Interface/interface/Py8InterfaceBase.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include <Pythia8/Pythia.h>
#include <Pythia8Plugins/HepMC2.h>

// foward declarations
namespace edm {
  class Event;
}

namespace CLHEP {
  class HepRandomEngine;
}

namespace gen {

  class Py8GunBase : public Py8InterfaceBase {
  public:
    Py8GunBase(edm::ParameterSet const& ps);
    ~Py8GunBase() override {}

    virtual bool residualDecay();  // common func
    bool initializeForInternalPartons() override;
    void finalizeEvent() override;
    void statistics() override;

    void setRandomEngine(CLHEP::HepRandomEngine* v) { p8SetRandomEngine(v); }
    std::vector<std::string> const& sharedResources() const { return p8SharedResources; }
    void evtGenDecay();

  protected:
    // (some of) PGun parameters
    //
    std::vector<int> fPartIDs;
    double fMinPhi;
    double fMaxPhi;

  private:
    static const std::vector<std::string> p8SharedResources;
  };

}  // namespace gen

#endif  // gen_BaseHadronizer_h

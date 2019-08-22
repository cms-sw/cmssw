#ifndef gen_TauolaInterface_TauolaInterfaceBase_h
#define gen_TauolaInterface_TauolaInterfaceBase_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "HepMC/GenEvent.h"
#include <vector>
#include "CLHEP/Random/RandomEngine.h"

// LHE Run
#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"
#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"

// LHE Event
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"

namespace gen {
  class TauolaInterfaceBase {
  public:
    TauolaInterfaceBase(){};
    TauolaInterfaceBase(const edm::ParameterSet&){};
    virtual ~TauolaInterfaceBase(){};

    virtual void SetDecayRandomEngine(CLHEP::HepRandomEngine* decayRandomEngine){};
    virtual void enablePolarization(){};
    virtual void disablePolarization(){};
    virtual void init(const edm::EventSetup&){};
    virtual const std::vector<int>& operatesOnParticles() { return fPDGs; }
    virtual HepMC::GenEvent* decay(HepMC::GenEvent* evt) { return evt; }
    virtual void statistics(){};
    virtual void setRandomEngine(CLHEP::HepRandomEngine* v) = 0;
    virtual void SetLHE(lhef::LHEEvent* l){};

  protected:
    std::vector<int> fPDGs;
  };
}  // namespace gen

#endif

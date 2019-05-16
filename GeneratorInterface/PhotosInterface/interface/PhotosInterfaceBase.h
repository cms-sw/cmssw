#ifndef gen_PhotosInterface_PhotosInterfaceBase_h
#define gen_PhotosInterface_PhotosInterfaceBase_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "HepMC/GenEvent.h"
#include <vector>
#include "CLHEP/Random/RandomEngine.h"
#include "HepMC/SimpleVector.h"
#include <string>
#include "CLHEP/Random/RandomEngine.h"

namespace gen {
  class PhotosInterfaceBase {
  public:
    PhotosInterfaceBase(){};
    PhotosInterfaceBase(const edm::ParameterSet&){};
    virtual ~PhotosInterfaceBase(){};

    virtual void SetDecayRandomEngine(CLHEP::HepRandomEngine* decayRandomEngine){};
    virtual void init() = 0;
    virtual const std::vector<std::string>& specialSettings() { return fSpecialSettings; }
    virtual HepMC::GenEvent* apply(HepMC::GenEvent* evt) { return evt; }
    virtual void avoidTauLeptonicDecays() = 0;
    ;
    virtual void configureOnlyFor(int) = 0;
    virtual void setRandomEngine(CLHEP::HepRandomEngine* decayRandomEngine) = 0;
    virtual void statistics(){};

  protected:
    std::vector<std::string> fSpecialSettings;
  };
}  // namespace gen

#endif

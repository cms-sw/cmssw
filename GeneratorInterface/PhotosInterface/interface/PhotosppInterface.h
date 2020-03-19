#ifndef gen_PhotosInterface_PhotosppInterface_h
#define gen_PhotosInterface_PhotosppInterface_h

// #include "HepPDT/ParticleDataTable.hh"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "HepMC/SimpleVector.h"
#include "GeneratorInterface/PhotosInterface/interface/PhotosInterfaceBase.h"

namespace HepMC {
  class GenEvent;
  class GenVertex;
}  // namespace HepMC

namespace gen {
  class PhotosppInterface : public PhotosInterfaceBase {
  public:
    // ctor & dtor
    PhotosppInterface(const edm::ParameterSet& pset);
    ~PhotosppInterface() override {}

    void init() override;
    const std::vector<std::string>& specialSettings() override { return fSpecialSettings; }
    HepMC::GenEvent* apply(HepMC::GenEvent*) override;
    void configureOnlyFor(int) override;
    void avoidTauLeptonicDecays() override {
      fAvoidTauLeptonicDecays = true;
      return;
    }
    bool isTauLeptonicDecay(HepMC::GenVertex*);
    void setRandomEngine(CLHEP::HepRandomEngine* decayRandomEngine) override;
    static double flat();
    void statistics() override;

  private:
    int fOnlyPDG;
    bool fAvoidTauLeptonicDecays;
    bool fIsInitialized;
    edm::ParameterSet* fPSet;

    static CLHEP::HepRandomEngine* fRandomEngine;
  };
}  // namespace gen

#endif

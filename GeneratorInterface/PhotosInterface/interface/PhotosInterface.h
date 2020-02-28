#ifndef gen_PhotosInterface_PhotosInterface_h
#define gen_PhotosInterface_PhotosInterface_h

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
  class PhotosInterface : public PhotosInterfaceBase {
  public:
    // ctor & dtor
    PhotosInterface();
    PhotosInterface(const edm::ParameterSet&);
    ~PhotosInterface() override;

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

  private:
    int fOnlyPDG;
    bool fAvoidTauLeptonicDecays;
    std::vector<int> fBarcodes;
    std::vector<int> fSecVtxStore;
    bool fIsInitialized;

    void applyToVertex(HepMC::GenEvent*, int);
    void applyToBranch(HepMC::GenEvent*, int);
    void attachParticles(HepMC::GenEvent*, HepMC::GenVertex*, int);

    static CLHEP::HepRandomEngine* fRandomEngine;
  };
}  // namespace gen

#endif

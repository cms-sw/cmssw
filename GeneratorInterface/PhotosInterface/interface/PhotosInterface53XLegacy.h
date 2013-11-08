#ifndef gen_PhotosInterface_PhotosInterface53XLegacy_h
#define gen_PhotosInterface_PhotosInterface53XLegacy_h

// #include "HepPDT/ParticleDataTable.hh"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "HepMC/SimpleVector.h"
#include "CLHEP/Random/RandomEngine.h"
#include "GeneratorInterface/PhotosInterface/interface/PhotosInterfaceBase.h"

namespace HepMC {
  class GenEvent;
  class GenVertex;
  class SimpleVector;
}



namespace gen {

  class PhotosInterface53XLegacy : public PhotosInterfaceBase {
  public:
      
    // ctor & dtor
    PhotosInterface53XLegacy();
    PhotosInterface53XLegacy(edm::ParameterSet const&);
    ~PhotosInterface53XLegacy() {}

    void SetDecayRandomEngine(CLHEP::HepRandomEngine* decayRandomEngine);
    void init();
    const std::vector<std::string>& specialSettings() { return fSpecialSettings; }
    HepMC::GenEvent* apply( HepMC::GenEvent* );
    void avoidTauLeptonicDecays() { fAvoidTauLeptonicDecays=true; return; }
      
  private:
            
    struct Scaling {
      HepMC::ThreeVector weights;
      int flag;
      Scaling( HepMC::ThreeVector vec, int flg )
	: weights(HepMC::ThreeVector(1.,1.,1)), flag(1) { weights=vec; flag=flg; }
    };

    int fOnlyPDG;
    bool fAvoidTauLeptonicDecays;
    std::vector<int> fBarcodes;
    bool fIsInitialized;
      
    void attachParticles( HepMC::GenEvent*, HepMC::GenVertex*, int );

  };

}

#endif

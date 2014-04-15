#ifndef gen_PhotosInterface_PhotosInterface_h
#define gen_PhotosInterface_PhotosInterface_h

// #include "HepPDT/ParticleDataTable.hh"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "HepMC/SimpleVector.h"
#include "GeneratorInterface/PhotosInterface/interface/PhotosInterfaceBase.h"


namespace HepMC 
{
class GenEvent;
class GenVertex;
}

namespace gen {
  class PhotosInterface : public PhotosInterfaceBase {
  public:
    
    // ctor & dtor
    PhotosInterface(); 
    PhotosInterface( const edm::ParameterSet& );
    ~PhotosInterface();

    void init();
    const std::vector<std::string>& specialSettings() { return fSpecialSettings; }
    HepMC::GenEvent* apply( HepMC::GenEvent* );
    void configureOnlyFor( int );
    void avoidTauLeptonicDecays() { fAvoidTauLeptonicDecays=true; return; }
    bool isTauLeptonicDecay( HepMC::GenVertex* );
    void setRandomEngine(CLHEP::HepRandomEngine* decayRandomEngine);
    static double flat();
    
  private: 
    struct Scaling {
      HepMC::ThreeVector weights;
      int                flag;
    Scaling( const HepMC::ThreeVector& vec, int flg )	
    : weights(HepMC::ThreeVector(1.,1.,1)), flag(1) { weights=vec; flag=flg; }
    };
    
    int                      fOnlyPDG;
    bool                     fAvoidTauLeptonicDecays;  
    std::vector<int>         fBarcodes;
    std::vector<int>         fSecVtxStore;
    bool                     fIsInitialized;
    
    void applyToVertex( HepMC::GenEvent*, int );
    void applyToBranch( HepMC::GenEvent*, int );
    void attachParticles( HepMC::GenEvent*, HepMC::GenVertex*, int );

    static CLHEP::HepRandomEngine* fRandomEngine;
    
  };
}

#endif

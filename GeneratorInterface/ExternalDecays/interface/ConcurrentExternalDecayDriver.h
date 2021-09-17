#ifndef gen_ConcurrentExternalDecayDriver_h
#define gen_ConcurrentExternalDecayDriver_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include <string>
#include <vector>

namespace HepMC {
  class GenEvent;
}

namespace CLHEP {
  class HepRandomEngine;
}

namespace lhef {
  class LHEEvent;
}

namespace gen {

  class EvtGenInterfaceBase;
  class TauolaInterfaceBase;
  class PhotosInterfaceBase;

  class ConcurrentExternalDecayDriver {
  public:
    // ctor & dtor
    ConcurrentExternalDecayDriver(const edm::ParameterSet&);
    ~ConcurrentExternalDecayDriver();

    void init(const edm::EventSetup&);

    const std::vector<int>& operatesOnParticles() { return fPDGs; }
    const std::vector<std::string>& specialSettings() { return fSpecialSettings; }

    HepMC::GenEvent* decay(HepMC::GenEvent* evt);
    HepMC::GenEvent* decay(HepMC::GenEvent* evt, lhef::LHEEvent* lheEvent);

    void statistics() const;

    void setRandomEngine(CLHEP::HepRandomEngine*);

  private:
    bool fIsInitialized;
    //std::unique_ptr<TauolaInterfaceBase> fTauolaInterface;
    //std::unique_ptr<EvtGenInterfaceBase> fEvtGenInterface;
    //std::unique_ptr<PhotosInterfaceBase> fPhotosInterface;
    std::vector<int> fPDGs;
    std::vector<std::string> fSpecialSettings;
  };

}  // namespace gen

#endif

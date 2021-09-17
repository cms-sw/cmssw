#ifndef gen_FortranCallback_h
#define gen_FortranCallback_h

#include "SimDataFormats/GeneratorProducts/interface/LHECommonBlocks.h"

#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"

namespace HepMC {
  class GenEvent;
}

namespace gen {

  class FortranCallback {
  public:
    static FortranCallback* getInstance();

    //void setLHEEvent(lhef::LHEEvent* lhev) { fPartonLevel = lhev; }
    void setLHERunInfo(lhef::LHERunInfo* lheri) { fRunBlock = lheri; }
    void setLHEEvent(lhef::LHEEvent* lhee) { fEventBlock = lhee; }

    void resetIterationsPerEvent() { fIterationsPerEvent = 0; }

    void fillHeader();
    void fillEvent();

    int getIterationsPerEvent() const { return fIterationsPerEvent; }

  private:
    // ctor

    FortranCallback();

    // data member(s)

    lhef::LHERunInfo* fRunBlock;
    lhef::LHEEvent* fEventBlock;
    int fIterationsPerEvent;

    static FortranCallback* fInstance;
  };

  // --** Implementation **---

  inline FortranCallback::FortranCallback()
      //   : fPartonLevel(0)
      : fRunBlock(nullptr), fEventBlock(nullptr), fIterationsPerEvent(0) {}

  inline void FortranCallback::fillHeader() {
    if (fRunBlock == nullptr)
      return;

    //const lhef::HEPRUP* heprup = &(fRunBlock->heprup());
    const lhef::HEPRUP* heprup = fRunBlock->getHEPRUP();

    lhef::CommonBlocks::fillHEPRUP(heprup);

    return;
  }

  inline void FortranCallback::fillEvent() {
    //if ( fPartonLevel == 0 ) return;
    //const lhef::HEPEUP* hepeup = fPartonLevel->getHEPEUP();

    if (fEventBlock == nullptr)
      return;

    const lhef::HEPEUP* hepeup = fEventBlock->getHEPEUP();

    if (fIterationsPerEvent++) {
      hepeup_.nup = 0;
      return;
    }

    lhef::CommonBlocks::fillHEPEUP(hepeup);

    return;
  }

}  // namespace gen

#endif

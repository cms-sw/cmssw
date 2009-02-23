// -*- C++ -*-
//
//

// class BaseHadronizer meant as base class for hadronizers
// implements a few common methods concerning communication with the
// gen::HadronizerFilter<...> and gen::GeneratorFilter<...> templates,
// mostly memory management regarding the GenEvent pointers and such

#ifndef gen_BaseHadronizer_h
#define gen_BaseHadronizer_h

#include <memory>

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"

namespace gen {

  class BaseHadronizer {
  public:
    BaseHadronizer() {}
    ~BaseHadronizer() {}

    const GenRunInfoProduct &getGenRunInfo() const { return genRunInfo_; }
    HepMC::GenEvent *getGenEvent() { return genEvent_.release(); }

    void resetEvent(HepMC::GenEvent *event) { genEvent_.reset(event); }

  protected:
    GenRunInfoProduct& runInfo() { return genRunInfo_; }
    std::auto_ptr<HepMC::GenEvent>& event() { return genEvent_; }

  private:
    GenRunInfoProduct               genRunInfo_;
    std::auto_ptr<HepMC::GenEvent>  genEvent_;
  };

} // namespace gen

#endif // gen_BaseHadronizer_h

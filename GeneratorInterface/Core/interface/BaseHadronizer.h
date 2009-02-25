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

#include <boost/shared_ptr.hpp>

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"

#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"

namespace gen {

  class BaseHadronizer {
  public:
    BaseHadronizer() {}
    ~BaseHadronizer() {}

    // GenRunInfo and GenEvent passing
    const GenRunInfoProduct &getGenRunInfo() const { return genRunInfo_; }
    HepMC::GenEvent *getGenEvent() { return genEvent_.release(); }

    void resetEvent(HepMC::GenEvent *event) { genEvent_.reset(event); }

    // LHERunInfo and LHEEvent passing
    const boost::shared_ptr<lhef::LHERunInfo> &getLHERunInfo() const { return lheRunInfo_; }

    void setLHERunInfo(lhef::LHERunInfo *runInfo) { lheRunInfo_.reset(runInfo); }
    void setLHEEvent(lhef::LHEEvent *event) { lheEvent_.reset(event); }

  protected:
    GenRunInfoProduct& runInfo() { return genRunInfo_; }
    std::auto_ptr<HepMC::GenEvent>& event() { return genEvent_; }

    lhef::LHEEvent* lheEvent() { return lheEvent_.get(); }
    lhef::LHERunInfo *lheRunInfo() { return lheRunInfo_.get(); }

  private:
    GenRunInfoProduct                   genRunInfo_;
    std::auto_ptr<HepMC::GenEvent>      genEvent_;

    boost::shared_ptr<lhef::LHERunInfo> lheRunInfo_;
    std::auto_ptr<lhef::LHEEvent>       lheEvent_;
  };

} // namespace gen

#endif // gen_BaseHadronizer_h

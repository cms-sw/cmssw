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
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// foward declarations
namespace edm {
  class Event;
}

namespace gen {

  class BaseHadronizer {
  public:
    BaseHadronizer( edm::ParameterSet const& ps );
    ~BaseHadronizer() {}

    // GenRunInfo and GenEvent passing
    GenRunInfoProduct &getGenRunInfo() { return genRunInfo_; }
    HepMC::GenEvent *getGenEvent() { return genEvent_.release(); }
    GenEventInfoProduct *getGenEventInfo() { return genEventInfo_.release(); }

    void resetEvent(HepMC::GenEvent *event) { genEvent_.reset(event); }
    void resetEventInfo(GenEventInfoProduct *eventInfo) { genEventInfo_.reset(eventInfo); }

    // LHERunInfo and LHEEvent passing
    const boost::shared_ptr<lhef::LHERunInfo> &getLHERunInfo() const { return lheRunInfo_; }

    void setLHERunInfo(lhef::LHERunInfo *runInfo) { lheRunInfo_.reset(runInfo); }
    void setLHEEvent(lhef::LHEEvent *event) { lheEvent_.reset(event); }

    // interface for accessing the EDM information from the hadronizer
    void setEDMEvent(edm::Event &event) { edmEvent_ = &event; }
    edm::Event &getEDMEvent() const { return *edmEvent_; }

  protected:
    GenRunInfoProduct& runInfo() { return genRunInfo_; }
    std::auto_ptr<HepMC::GenEvent>& event() { return genEvent_; }
    std::auto_ptr<GenEventInfoProduct>& eventInfo() { return genEventInfo_; }

    lhef::LHEEvent* lheEvent() { return lheEvent_.get(); }
    lhef::LHERunInfo *lheRunInfo() { return lheRunInfo_.get(); }

  private:
    GenRunInfoProduct                   genRunInfo_;
    std::auto_ptr<HepMC::GenEvent>      genEvent_;
    std::auto_ptr<GenEventInfoProduct>  genEventInfo_;

    boost::shared_ptr<lhef::LHERunInfo> lheRunInfo_;
    std::auto_ptr<lhef::LHEEvent>       lheEvent_;

    edm::Event                          *edmEvent_;
  };

} // namespace gen

#endif // gen_BaseHadronizer_h

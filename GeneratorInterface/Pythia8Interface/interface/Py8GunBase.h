#ifndef gen_Py8GunBase_h
#define gen_Py8GunBase_h

#include <memory>

#include <boost/shared_ptr.hpp>

#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "GeneratorInterface/Pythia8Interface/interface/Py8InterfaceBase.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <Pythia.h>
#include <HepMCInterface.h>

// foward declarations
namespace edm {
  class Event;
}

namespace gen {

  class Py8GunBase : public Py8InterfaceBase {
  public:
    Py8GunBase( edm::ParameterSet const& ps );
    ~Py8GunBase() {}

    // GenRunInfo and GenEvent passing
    GenRunInfoProduct &getGenRunInfo() { return genRunInfo_; }
    HepMC::GenEvent *getGenEvent() { return genEvent_.release(); }
    GenEventInfoProduct *getGenEventInfo() { return genEventInfo_.release(); }

    void resetEvent(HepMC::GenEvent *event) { genEvent_.reset(event); }
    void resetEventInfo(GenEventInfoProduct *eventInfo) { genEventInfo_.reset(eventInfo); }

    // interface for accessing the EDM information from the hadronizer
    void setEDMEvent(edm::Event &event) { edmEvent_ = &event; }
    edm::Event &getEDMEvent() const { return *edmEvent_; }
    virtual bool select(HepMC::GenEvent*) const { return true;}
    
    virtual bool residualDecay(); // common func
    bool initializeForInternalPartons();
    void finalizeEvent(); 
    void statistics();

  protected:
    GenRunInfoProduct& runInfo() { return genRunInfo_; }
    std::auto_ptr<HepMC::GenEvent>& event() { return genEvent_; }
    std::auto_ptr<GenEventInfoProduct>& eventInfo() { return genEventInfo_; }
        
    // (some of) PGun parameters
    //
    std::vector<int> fPartIDs ;
    double           fMinPhi ;
    double           fMaxPhi ;
    
  private:
    GenRunInfoProduct                   genRunInfo_;
    std::auto_ptr<HepMC::GenEvent>      genEvent_;
    std::auto_ptr<GenEventInfoProduct>  genEventInfo_;

    edm::Event                          *edmEvent_;

  };

} // namespace gen

#endif // gen_Py8GunBase_h

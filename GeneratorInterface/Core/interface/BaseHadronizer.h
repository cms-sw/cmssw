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
#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenLumiInfoHeader.h"

#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"

#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "CLHEP/Random/RandomEngine.h"


// foward declarations
namespace edm {
  class Event;
}

namespace CLHEP {
  class HepRandomEngine;
}

namespace gen {

  class BaseHadronizer {
  public:
    BaseHadronizer( edm::ParameterSet const& ps );
    virtual ~BaseHadronizer() noexcept (false) {}

    // GenRunInfo and GenEvent passing
    GenRunInfoProduct &getGenRunInfo() { return genRunInfo_; }
    HepMC::GenEvent *getGenEvent() { return genEvent_.release(); }
    GenEventInfoProduct *getGenEventInfo() { return genEventInfo_.release(); }
    virtual GenLumiInfoHeader *getGenLumiInfoHeader() const;
    
    void resetEvent(HepMC::GenEvent *event) { genEvent_.reset(event); }
    void resetEventInfo(GenEventInfoProduct *eventInfo) { genEventInfo_.reset(eventInfo); }

    // LHERunInfo and LHEEvent passing
    const boost::shared_ptr<lhef::LHERunInfo> &getLHERunInfo() const { return lheRunInfo_; }

    void setLHERunInfo(lhef::LHERunInfo *runInfo) { lheRunInfo_.reset(runInfo); }
    void setLHEEvent(lhef::LHEEvent *event) { lheEvent_.reset(event); }

    // interface for accessing the EDM information from the hadronizer
    void setEDMEvent(edm::Event &event) { edmEvent_ = &event; }
    edm::Event &getEDMEvent() const { return *edmEvent_; }
    virtual bool select(HepMC::GenEvent*) const { return true;}

    void setRandomEngine(CLHEP::HepRandomEngine* v) { doSetRandomEngine(v); }

    std::vector<std::string> const& sharedResources() const { return doSharedResources(); }
    
    int randomIndex() const { return randomIndex_; }
    const std::string &randomInitConfigDescription() const { return randomInitConfigDescriptions_[randomIndex_]; }
    const std::string &gridpackPath() const { return gridpackPaths_[std::max(randomIndex_,0)]; }
    
    void randomizeIndex(edm::LuminosityBlock const& lumi, CLHEP::HepRandomEngine* rengine);
    void generateLHE(edm::LuminosityBlock const& lumi, CLHEP::HepRandomEngine* rengine, unsigned int ncpu);
    void cleanLHE();

  protected:
    GenRunInfoProduct& runInfo() { return genRunInfo_; }
    std::unique_ptr<HepMC::GenEvent>& event() { return genEvent_; }
    std::unique_ptr<GenEventInfoProduct>& eventInfo() { return genEventInfo_; }

    lhef::LHEEvent* lheEvent() { return lheEvent_.get(); }
    lhef::LHERunInfo *lheRunInfo() { return lheRunInfo_.get(); }
    int randomIndex_;
    std::string lheFile_;

  private:

    virtual void doSetRandomEngine(CLHEP::HepRandomEngine* v) { }

    virtual std::vector<std::string> const& doSharedResources() const { return theSharedResources; }

    GenRunInfoProduct                   genRunInfo_;
    std::unique_ptr<HepMC::GenEvent>      genEvent_;
    std::unique_ptr<GenEventInfoProduct>  genEventInfo_;

    boost::shared_ptr<lhef::LHERunInfo> lheRunInfo_;
    std::unique_ptr<lhef::LHEEvent>       lheEvent_;

    edm::Event                          *edmEvent_;

    static const std::vector<std::string> theSharedResources;

    std::vector<double> randomInitWeights_;
    std::vector<std::string> randomInitConfigDescriptions_;
    std::vector<std::string> gridpackPaths_;
        
  };

} // namespace gen

#endif // gen_BaseHadronizer_h

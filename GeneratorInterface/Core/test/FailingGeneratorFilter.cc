// -*- C++ -*-
//
// Package:     GeneratorInterface/Core
// Class  :     FailingGeneratorFilter
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Wed, 08 Sep 2021 15:58:11 GMT
//

// system include files

// user include files
#include "GeneratorInterface/Core/interface/BaseHadronizer.h"
#include "GeneratorInterface/Core/interface/GeneratorFilter.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <csignal>
namespace test {
  class FailingHad {
  public:
    FailingHad(const edm::ParameterSet& iPSet)
        : failAt_(iPSet.getParameter<int>("failAt")), failureType_(iPSet.getParameter<int>("failureType")) {}

    std::vector<std::string> sharedResources() const {
      if (failAt_ == 0) {
        fail("Constructor");
      }
      return {};
    }

    void setEDMEvent(edm::Event&) { event_ = std::make_unique<HepMC::GenEvent>(); }

    bool generatePartonsAndHadronize() const {
      if (failAt_ == 2) {
        fail("Event");
      }
      return true;
    }
    bool decay() const { return true; }

    unsigned int getVHepMC() { return ivhepmc; }
    std::unique_ptr<HepMC::GenEvent> getGenEvent() { return std::move(event_); }
    std::unique_ptr<HepMC3::GenEvent> getGenEvent3() { return std::move(event3_); }

    bool select(HepMC::GenEvent*) const { return true; }
    void resetEvent(std::unique_ptr<HepMC::GenEvent> iEvent) { event_ = std::move(iEvent); }
    void resetEvent3(std::unique_ptr<HepMC3::GenEvent> iEvent) { event3_ = std::move(iEvent); }
    bool residualDecay() const { return true; }
    void finalizeEvent() const {}
    std::unique_ptr<GenEventInfoProduct> getGenEventInfo() const { return std::make_unique<GenEventInfoProduct>(); }
    std::unique_ptr<GenEventInfoProduct3> getGenEventInfo3() const { return std::make_unique<GenEventInfoProduct3>(); }

    //caled at endRunProduce, endLumiProduce
    void statistics() const {}
    GenRunInfoProduct getGenRunInfo() const { return GenRunInfoProduct(); }

    //called begin Lumi Produce
    template <typename T>
    void randomizeIndex(edm::LuminosityBlock const&, T&&) const {
      if (failAt_ == 1) {
        fail("BeginLumi");
      }
    }
    template <typename T>
    void generateLHE(edm::LuminosityBlock const&, T, unsigned int) const {}

    bool readSettings(int) const { return true; }
    std::string classname() const { return ""; }
    template <typename T>
    bool declareStableParticles(T&&) const {
      return true;
    }
    template <typename T>
    bool declareSpecialSettings(T&&) const {
      return true;
    }
    bool initializeForInternalPartons() const { return true; }
    std::unique_ptr<GenLumiInfoHeader> getGenLumiInfoHeader() const { return std::make_unique<GenLumiInfoHeader>(); }

    //called end lumi
    void cleanLHE() const {};

    template <typename T>
    void setRandomEngine(T&&) const {}

  private:
    void fail(std::string const& iName) const {
      switch (failureType_) {
        case 0: {
          throw cms::Exception(iName);
        }
        case 1: {
          std::raise(SIGSEGV);
          break;
        }
        case 2: {
          std::terminate();
          break;
        }
        default: {
          std::exit(-1);
        }
      }
    }
    std::unique_ptr<HepMC::GenEvent> event_;
    std::unique_ptr<HepMC3::GenEvent> event3_;
    int failAt_;
    int failureType_;
    unsigned int ivhepmc = 2;
  };

  class DummyDec {
  public:
    DummyDec(const edm::ParameterSet&, edm::ConsumesCollector) {}
    std::vector<std::string> sharedResources() const { return {}; }

    HepMC::GenEvent* decay(HepMC::GenEvent const*) { return nullptr; }
    void statistics() const {}

    void init(const edm::EventSetup&) const {}
    bool operatesOnParticles() const { return false; }
    bool specialSettings() const { return false; }

    template <typename T>
    void setRandomEngine(T&&) const {}
  };
}  // namespace test

using FailingGeneratorFilter = edm::GeneratorFilter<test::FailingHad, test::DummyDec>;

DEFINE_FWK_MODULE(FailingGeneratorFilter);

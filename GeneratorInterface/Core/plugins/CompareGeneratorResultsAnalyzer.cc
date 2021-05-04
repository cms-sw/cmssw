#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenLumiInfoHeader.h"
#include "SimDataFormats/GeneratorProducts/interface/GenLumiInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include <sstream>
#include <iostream>

namespace cgra {
  struct DummyCache {};
};  // namespace cgra

class CompareGeneratorResultsAnalyzer
    : public edm::global::EDAnalyzer<edm::RunCache<cgra::DummyCache>, edm::LuminosityBlockCache<cgra::DummyCache>> {
public:
  CompareGeneratorResultsAnalyzer(edm::ParameterSet const&);

  std::shared_ptr<cgra::DummyCache> globalBeginRun(edm::Run const&, edm::EventSetup const&) const override;
  void globalEndRun(edm::Run const&, edm::EventSetup const&) const override;

  std::shared_ptr<cgra::DummyCache> globalBeginLuminosityBlock(edm::LuminosityBlock const&,
                                                               edm::EventSetup const&) const override;
  void globalEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) const override;

  void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;

private:
  std::string mod1_;
  std::string mod2_;

  edm::EDGetTokenT<GenEventInfoProduct> evToken1_;
  edm::EDGetTokenT<GenEventInfoProduct> evToken2_;

  edm::EDGetTokenT<edm::HepMCProduct> hepMCToken1_;
  edm::EDGetTokenT<edm::HepMCProduct> hepMCToken2_;

  edm::EDGetTokenT<GenLumiInfoHeader> lumiHeaderToken1_;
  edm::EDGetTokenT<GenLumiInfoHeader> lumiHeaderToken2_;

  edm::EDGetTokenT<GenLumiInfoProduct> lumiProductToken1_;
  edm::EDGetTokenT<GenLumiInfoProduct> lumiProductToken2_;

  edm::EDGetTokenT<GenRunInfoProduct> runProductToken1_;
  edm::EDGetTokenT<GenRunInfoProduct> runProductToken2_;

  bool allowXSecDifferences_;
};

CompareGeneratorResultsAnalyzer::CompareGeneratorResultsAnalyzer(edm::ParameterSet const& iPSet)
    : mod1_{iPSet.getUntrackedParameter<std::string>("module1")},
      mod2_{iPSet.getUntrackedParameter<std::string>("module2")},
      evToken1_{consumes<GenEventInfoProduct>(mod1_)},
      evToken2_{consumes<GenEventInfoProduct>(mod2_)},
      hepMCToken1_{consumes<edm::HepMCProduct>(edm::InputTag(mod1_, "unsmeared"))},
      hepMCToken2_{consumes<edm::HepMCProduct>(edm::InputTag(mod2_, "unsmeared"))},
      lumiHeaderToken1_{consumes<GenLumiInfoHeader, edm::InLumi>(mod1_)},
      lumiHeaderToken2_{consumes<GenLumiInfoHeader, edm::InLumi>(mod2_)},
      lumiProductToken1_{consumes<GenLumiInfoProduct, edm::InLumi>(mod1_)},
      lumiProductToken2_{consumes<GenLumiInfoProduct, edm::InLumi>(mod2_)},
      runProductToken1_{consumes<GenRunInfoProduct, edm::InRun>(mod1_)},
      runProductToken2_{consumes<GenRunInfoProduct, edm::InRun>(mod2_)},
      allowXSecDifferences_{iPSet.getUntrackedParameter<bool>("allowXSecDifferences", false)} {}

std::shared_ptr<cgra::DummyCache> CompareGeneratorResultsAnalyzer::globalBeginRun(edm::Run const&,
                                                                                  edm::EventSetup const&) const {
  return std::shared_ptr<cgra::DummyCache>();
}

void CompareGeneratorResultsAnalyzer::globalEndRun(edm::Run const& iRun, edm::EventSetup const&) const {
  auto const& prod1 = iRun.get(runProductToken1_);
  auto const& prod2 = iRun.get(runProductToken2_);

  if (not prod1.isProductEqual(prod2)) {
    throw cms::Exception("ComparisonFailure") << "The GenRunInfoProducts are different";
  }
}

std::shared_ptr<cgra::DummyCache> CompareGeneratorResultsAnalyzer::globalBeginLuminosityBlock(
    edm::LuminosityBlock const& iLumi, edm::EventSetup const&) const {
  auto const& prod1 = iLumi.get(lumiHeaderToken1_);
  auto const& prod2 = iLumi.get(lumiHeaderToken2_);

  if (prod1.randomConfigIndex() != prod2.randomConfigIndex()) {
    throw cms::Exception("ComparisonFailure") << "The GenLumiInfoHeaders have different randomConfigIndex "
                                              << prod1.randomConfigIndex() << " " << prod2.randomConfigIndex();
  }

  if (prod1.configDescription() != prod2.configDescription()) {
    throw cms::Exception("ComparisonFailure") << "The GenLumiInfoHeaders have different configDescription "
                                              << prod1.configDescription() << " " << prod2.configDescription();
  }

  if (prod1.lheHeaders().size() != prod2.lheHeaders().size()) {
    throw cms::Exception("ComparisonFailure") << "The GenLumiInfoHeaders have different lheHeaders "
                                              << prod1.lheHeaders().size() << " " << prod2.lheHeaders().size();
  }

  if (prod1.weightNames().size() != prod2.weightNames().size()) {
    throw cms::Exception("ComparisonFailure") << "The GenLumiInfoHeaders have different weightNames "
                                              << prod1.weightNames().size() << " " << prod2.weightNames().size();
  }

  return std::shared_ptr<cgra::DummyCache>();
}

namespace {
  void compare(size_t iIndex,
               GenLumiInfoProduct::ProcessInfo const& p1,
               GenLumiInfoProduct::ProcessInfo const& p2,
               bool allowXSecDifferences) {
    if (p1.process() != p2.process()) {
      throw cms::Exception("ComparisonFailure") << "The GenLumiInfoProducts have different getProcessInfos()[" << iIndex
                                                << "] process " << p1.process() << " " << p2.process();
    }

    if (p1.nPassPos() != p2.nPassPos()) {
      throw cms::Exception("ComparisonFailure") << "The GenLumiInfoProducts have different getProcessInfos()[" << iIndex
                                                << "] nPassPos " << p1.nPassPos() << " " << p2.nPassPos();
    }

    if (p1.nPassNeg() != p2.nPassNeg()) {
      throw cms::Exception("ComparisonFailure") << "The GenLumiInfoProducts have different getProcessInfos()[" << iIndex
                                                << "] nPassNeg " << p1.nPassNeg() << " " << p2.nPassNeg();
    }

    if (p1.nTotalPos() != p2.nTotalPos()) {
      throw cms::Exception("ComparisonFailure") << "The GenLumiInfoProducts have different getProcessInfos()[" << iIndex
                                                << "] nTotalPos " << p1.nTotalPos() << " " << p2.nTotalPos();
    }

    if (p1.nTotalNeg() != p2.nTotalNeg()) {
      throw cms::Exception("ComparisonFailure") << "The GenLumiInfoProducts have different getProcessInfos()[" << iIndex
                                                << "] nTotalNeg " << p1.nTotalNeg() << " " << p2.nTotalNeg();
    }

    if (p1.lheXSec().error() != p2.lheXSec().error()) {
      if (allowXSecDifferences) {
        edm::LogWarning("ComparisonFailure")
            << "The GenLumiInfoProducts have different getProcessInfos()[" << iIndex << "] lheXSec.error "
            << p1.lheXSec().error() << " " << p2.lheXSec().error();
      } else {
        throw cms::Exception("ComparisonFailure")
            << "The GenLumiInfoProducts have different getProcessInfos()[" << iIndex << "] lheXSec.error "
            << p1.lheXSec().error() << " " << p2.lheXSec().error();
      }
    }

    if (p1.lheXSec().value() != p2.lheXSec().value()) {
      if (allowXSecDifferences) {
        //throw cms::Exception("ComparisonFailure")
        edm::LogWarning("ComparisonFailure")
            << "The GenLumiInfoProducts have different getProcessInfos()[" << iIndex << "] lheXSec.value "
            << p1.lheXSec().value() << " " << p2.lheXSec().value();
      } else {
        throw cms::Exception("ComparisonFailure")
            << "The GenLumiInfoProducts have different getProcessInfos()[" << iIndex << "] lheXSec.value "
            << p1.lheXSec().value() << " " << p2.lheXSec().value();
      }
    }

    if (p1.tried().n() != p2.tried().n()) {
      throw cms::Exception("ComparisonFailure") << "The GenLumiInfoProducts have different getProcessInfos()[" << iIndex
                                                << "] tried.n " << p1.tried().n() << " " << p2.tried().n();
    }

    if (p1.tried().sum() != p2.tried().sum()) {
      throw cms::Exception("ComparisonFailure") << "The GenLumiInfoProducts have different getProcessInfos()[" << iIndex
                                                << "] tried.sum " << p1.tried().sum() << " " << p2.tried().sum();
    }

    if (p1.tried().sum2() != p2.tried().sum2()) {
      throw cms::Exception("ComparisonFailure") << "The GenLumiInfoProducts have different getProcessInfos()[" << iIndex
                                                << "] tried.sum2 " << p1.tried().sum2() << " " << p2.tried().sum2();
    }

    if (p1.selected().n() != p2.selected().n()) {
      throw cms::Exception("ComparisonFailure") << "The GenLumiInfoProducts have different getProcessInfos()[" << iIndex
                                                << "] selected.n " << p1.selected().n() << " " << p2.selected().n();
    }

    if (p1.selected().sum() != p2.selected().sum()) {
      throw cms::Exception("ComparisonFailure")
          << "The GenLumiInfoProducts have different getProcessInfos()[" << iIndex << "] selected.sum "
          << p1.selected().sum() << " " << p2.selected().sum();
    }

    if (p1.selected().sum2() != p2.selected().sum2()) {
      throw cms::Exception("ComparisonFailure")
          << "The GenLumiInfoProducts have different getProcessInfos()[" << iIndex << "] selected.sum2 "
          << p1.selected().sum2() << " " << p2.selected().sum2();
    }

    if (p1.killed().n() != p2.killed().n()) {
      throw cms::Exception("ComparisonFailure") << "The GenLumiInfoProducts have different getProcessInfos()[" << iIndex
                                                << "] killed.n " << p1.killed().n() << " " << p2.killed().n();
    }

    if (p1.killed().sum() != p2.killed().sum()) {
      throw cms::Exception("ComparisonFailure") << "The GenLumiInfoProducts have different getProcessInfos()[" << iIndex
                                                << "] killed sum " << p1.killed().sum() << " " << p2.killed().sum();
    }

    if (p1.killed().sum2() != p2.killed().sum2()) {
      throw cms::Exception("ComparisonFailure") << "The GenLumiInfoProducts have different getProcessInfos()[" << iIndex
                                                << "] killed.sum2 " << p1.killed().sum2() << " " << p2.killed().sum2();
    }

    if (p1.accepted().n() != p2.accepted().n()) {
      throw cms::Exception("ComparisonFailure") << "The GenLumiInfoProducts have different getProcessInfos()[" << iIndex
                                                << "] accepted.n " << p1.accepted().n() << " " << p2.accepted().n();
    }

    if (p1.accepted().sum() != p2.accepted().sum()) {
      throw cms::Exception("ComparisonFailure")
          << "The GenLumiInfoProducts have different getProcessInfos()[" << iIndex << "] accepted.sum "
          << p1.accepted().sum() << " " << p2.accepted().sum();
    }

    if (p1.accepted().sum2() != p2.accepted().sum2()) {
      throw cms::Exception("ComparisonFailure")
          << "The GenLumiInfoProducts have different getProcessInfos()[" << iIndex << "] accepted.sum2 "
          << p1.accepted().sum2() << " " << p2.accepted().sum2();
    }

    if (p1.acceptedBr().n() != p2.acceptedBr().n()) {
      throw cms::Exception("ComparisonFailure")
          << "The GenLumiInfoProducts have different getProcessInfos()[" << iIndex << "] acceptedBr.n "
          << p1.acceptedBr().n() << " " << p2.acceptedBr().n();
    }

    if (p1.acceptedBr().sum() != p2.acceptedBr().sum()) {
      throw cms::Exception("ComparisonFailure")
          << "The GenLumiInfoProducts have different getProcessInfos()[" << iIndex << "] acceptedZBr.sum "
          << p1.acceptedBr().sum() << " " << p2.acceptedBr().sum();
    }

    if (p1.acceptedBr().sum2() != p2.acceptedBr().sum2()) {
      throw cms::Exception("ComparisonFailure")
          << "The GenLumiInfoProducts have different getProcessInfos()[" << iIndex << "] acceptedBr.sum2 "
          << p1.acceptedBr().sum2() << " " << p2.acceptedBr().sum2();
    }
  }
}  // namespace

void CompareGeneratorResultsAnalyzer::globalEndLuminosityBlock(edm::LuminosityBlock const& iLumi,
                                                               edm::EventSetup const&) const {
  auto const& prod1 = iLumi.get(lumiProductToken1_);
  auto const& prod2 = iLumi.get(lumiProductToken2_);

  if (not prod1.isProductEqual(prod2)) {
    if (prod1.getHEPIDWTUP() != prod1.getHEPIDWTUP()) {
      throw cms::Exception("ComparisonFailure") << "The GenLumiInfoProducts have different getHEPIDWTUP "
                                                << prod1.getHEPIDWTUP() << " " << prod2.getHEPIDWTUP();
    }

    if (prod1.getProcessInfos().size() != prod2.getProcessInfos().size()) {
      throw cms::Exception("ComparisonFailure")
          << "The GenLumiInfoHeaders have different getProcessInfos " << prod1.getProcessInfos().size() << " "
          << prod2.getProcessInfos().size();
    }

    for (size_t i = 0; i < prod1.getProcessInfos().size(); ++i) {
      compare(i, prod1.getProcessInfos()[i], prod2.getProcessInfos()[i], allowXSecDifferences_);
    }

    if (not allowXSecDifferences_) {
      throw cms::Exception("ComparisionFailure") << "The GenLumiInfoProducts are different";
    }
  }
}

namespace {
  void compare(GenEventInfoProduct const& prod1, GenEventInfoProduct const& prod2) {
    if (prod1.weights().size() != prod2.weights().size()) {
      throw cms::Exception("ComparisonFailure") << "The GenEventInfoProducts have different weights "
                                                << prod1.weights().size() << " " << prod2.weights().size();
    }

    if (prod1.binningValues().size() != prod2.binningValues().size()) {
      throw cms::Exception("ComparisonFailure") << "The GenEventInfoProducts have different binningValues "
                                                << prod1.binningValues().size() << " " << prod2.binningValues().size();
    }

    if (prod1.DJRValues().size() != prod2.DJRValues().size()) {
      throw cms::Exception("ComparisonFailure") << "The GenEventInfoProducts have different DJRValues "
                                                << prod1.DJRValues().size() << " " << prod2.DJRValues().size();
    }

    if (prod1.signalProcessID() != prod2.signalProcessID()) {
      throw cms::Exception("ComparisonFailure") << "The GenEventInfoProducts have different signalProcessID "
                                                << prod1.signalProcessID() << " " << prod2.signalProcessID();
    }

    if (prod1.qScale() != prod2.qScale()) {
      throw cms::Exception("ComparisonFailure")
          << "The GenEventInfoProducts have different qScale " << prod1.qScale() << " " << prod2.qScale();
    }

    if (prod1.alphaQCD() != prod2.alphaQCD()) {
      throw cms::Exception("ComparisonFailure")
          << "The GenEventInfoProducts have different alphaQCD " << prod1.alphaQCD() << " " << prod2.alphaQCD();
    }

    if (prod1.alphaQED() != prod2.alphaQED()) {
      throw cms::Exception("ComparisonFailure")
          << "The GenEventInfoProducts have different alphaQED " << prod1.alphaQED() << " " << prod2.alphaQED();
    }

    if (prod1.nMEPartons() != prod2.nMEPartons()) {
      throw cms::Exception("ComparisonFailure")
          << "The GenEventInfoProducts have different nMEPartons " << prod1.nMEPartons() << " " << prod2.nMEPartons();
    }

    if (prod1.nMEPartonsFiltered() != prod2.nMEPartonsFiltered()) {
      throw cms::Exception("ComparisonFailure") << "The GenEventInfoProducts have different nMEPartonsFiltered "
                                                << prod1.nMEPartonsFiltered() << " " << prod2.nMEPartonsFiltered();
    }
  }

  void compare(HepMC::GenEvent const& prod1, HepMC::GenEvent const& prod2) {
    if (prod1.signal_process_id() != prod2.signal_process_id()) {
      throw cms::Exception("ComparisonFailure") << "The HepMCProducts have different signal_process_id "
                                                << prod1.signal_process_id() << " " << prod2.signal_process_id();
    }

    if (prod1.vertices_size() != prod2.vertices_size()) {
      throw cms::Exception("ComparisonFailure") << "The HepMCProducts have different vertices_size() "
                                                << prod1.vertices_size() << " " << prod2.vertices_size();
    }

    if (prod1.particles_size() != prod2.particles_size()) {
      throw cms::Exception("ComparisonFailure") << "The HepMCProducts have different particles_size() "
                                                << prod1.particles_size() << " " << prod2.particles_size();
    }
  }
}  // namespace

void CompareGeneratorResultsAnalyzer::analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const&) const {
  auto const& prod1 = iEvent.get(evToken1_);
  auto const& prod2 = iEvent.get(evToken2_);

  compare(prod1, prod2);

  auto const& hepmc1 = iEvent.get(hepMCToken1_);
  auto const& hepmc2 = iEvent.get(hepMCToken2_);

  compare(hepmc1.getHepMCData(), hepmc2.getHepMCData());
}

DEFINE_FWK_MODULE(CompareGeneratorResultsAnalyzer);

#include <memory>
#include <sstream>
#include <fstream>

#include <HepMC3/GenEvent.h>
//#include <HepMC/IO_BaseClass.h>
#include "HepMC3/Print.h"

#include <ThePEG/Repository/Repository.h>
#include <ThePEG/EventRecord/Event.h>
#include <ThePEG/Config/ThePEG.h>
#include <ThePEG/LesHouches/LesHouchesReader.h>

#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMC3Product.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct3.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"

#include "GeneratorInterface/Core/interface/BaseHadronizer.h"
#include "GeneratorInterface/Core/interface/GeneratorFilter.h"
#include "GeneratorInterface/Core/interface/HadronizerFilter.h"

#include "GeneratorInterface/LHEInterface/interface/LHEProxy.h"

#include "GeneratorInterface/Herwig7Interface/interface/Herwig7HepMC3Interface.h"

#include <Herwig/API/HerwigAPI.h>
#include "CLHEP/Random/RandomEngine.h"

namespace CLHEP {
  class HepRandomEngine;
}

class Herwig7HepMC3Hadronizer : public Herwig7HepMC3Interface, public gen::BaseHadronizer {
public:
  Herwig7HepMC3Hadronizer(const edm::ParameterSet& params);
  ~Herwig7HepMC3Hadronizer() override;

  bool readSettings(int) { return true; }
  bool initializeForInternalPartons();
  bool initializeForExternalPartons();
  bool declareStableParticles(const std::vector<int>& pdgIds);
  bool declareSpecialSettings(const std::vector<std::string>) { return true; }

  void statistics();

  bool generatePartonsAndHadronize();
  bool hadronize();
  bool decay();
  bool residualDecay();
  void finalizeEvent();

  const char* classname() const { return "Herwig7HepMC3Hadronizer"; }
  std::unique_ptr<GenLumiInfoHeader> getGenLumiInfoHeader() const override;
  void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
  void randomizeIndex(edm::LuminosityBlock const& lumi, CLHEP::HepRandomEngine* rengine);

private:
  void doSetRandomEngine(CLHEP::HepRandomEngine* v) override { setPEGRandomEngine(v); }

  unsigned int eventsToPrint;

  ThePEG::EventPtr thepegEvent;
  bool haveEvt = false;

  std::shared_ptr<lhef::LHEProxy> proxy_;
  const std::string handlerDirectory_;
  edm::ParameterSet paramSettings;
  const std::string runFileName;

  unsigned int firstLumiBlock = 0;
  unsigned int currentLumiBlock = 0;
};

Herwig7HepMC3Hadronizer::Herwig7HepMC3Hadronizer(const edm::ParameterSet& pset)
    : Herwig7HepMC3Interface(pset),
      BaseHadronizer(pset),
      eventsToPrint(pset.getUntrackedParameter<unsigned int>("eventsToPrint", 0)),
      handlerDirectory_(pset.getParameter<std::string>("eventHandlers")),
      runFileName(pset.getParameter<std::string>("run")) {
  ivhepmc = 3;
  initRepository(pset);
  paramSettings = pset;
}

Herwig7HepMC3Hadronizer::~Herwig7HepMC3Hadronizer() {}

bool Herwig7HepMC3Hadronizer::initializeForInternalPartons() {
  if (currentLumiBlock == firstLumiBlock) {
    std::ifstream runFile(runFileName + ".run");
    if (runFile.fail())  //required for showering of LHE files
    {
      initRepository(paramSettings);
    }
    if (!initGenerator()) {
      edm::LogInfo("Generator|Herwig7HepMC3Hadronizer") << "No run step for Herwig chosen. Program will be aborted.";
      exit(0);
    }
  }
  return true;
}

bool Herwig7HepMC3Hadronizer::initializeForExternalPartons() {
  if (currentLumiBlock == firstLumiBlock) {
    std::ifstream runFile(runFileName + ".run");
    if (runFile.fail())  //required for showering of LHE files
    {
      initRepository(paramSettings);
    }
    if (!initGenerator()) {
      edm::LogInfo("Generator|Herwig7HepMC3Hadronizer") << "No run step for Herwig chosen. Program will be aborted.";
      exit(0);
    }
  }
  return true;
}

bool Herwig7HepMC3Hadronizer::declareStableParticles(const std::vector<int>& pdgIds) { return false; }

void Herwig7HepMC3Hadronizer::statistics() {
  if (eg_) {
    runInfo().setInternalXSec(
        GenRunInfoProduct::XSec(eg_->integratedXSec() / ThePEG::picobarn, eg_->integratedXSecErr() / ThePEG::picobarn));
  }
}

bool Herwig7HepMC3Hadronizer::generatePartonsAndHadronize() {
  edm::LogInfo("Generator|Herwig7HepMC3Hadronizer") << "Start production";

  try {
    thepegEvent = eg_->shoot();
  } catch (std::exception& exc) {
    edm::LogWarning("Generator|Herwig7HepMC3Hadronizer")
        << "EGPtr::shoot() thrown an exception, event skipped: " << exc.what();
    return false;
  }

  if (!thepegEvent) {
    edm::LogWarning("Generator|Herwig7HepMC3Hadronizer") << "thepegEvent not initialized";
    return false;
  }

  event3() = convert(thepegEvent);
  if (!event3().get()) {
    edm::LogWarning("Generator|Herwig7HepMC3Hadronizer") << "genEvent not initialized";
    return false;
  }

  return true;
}

bool Herwig7HepMC3Hadronizer::hadronize() {
  if (!haveEvt) {
    try {
      thepegEvent = eg_->shoot();
      haveEvt = true;
    } catch (std::exception& exc) {
      edm::LogWarning("Generator|Herwig7HepMC3Hadronizer")
          << "EGPtr::shoot() thrown an exception, event skipped: " << exc.what();
      return false;
    }
  }
  int evtnum = lheEvent()->evtnum();
  if (evtnum == -1) {
    edm::LogError("Generator|Herwig7HepMC3Hadronizer")
        << "Event number not set in lhe file, needed for correctly aligning Herwig and LHE events!";
    return false;
  }
  if (thepegEvent->number() < evtnum) {
    edm::LogError("Herwig7 interface") << "Herwig does not seem to be generating events in order, did you set "
                                          "/Herwig/EventHandlers/FxFxLHReader:AllowedToReOpen Yes?";
    return false;
  } else if (thepegEvent->number() == evtnum) {
    haveEvt = false;
    if (!thepegEvent) {
      edm::LogWarning("Generator|Herwig7HepMC3Hadronizer") << "thepegEvent not initialized";
      return false;
    }

    event3() = convert(thepegEvent);
    if (!event3().get()) {
      edm::LogWarning("Generator|Herwig7HepMC3Hadronizer") << "genEvent not initialized";
      return false;
    }
    return true;
  }
  edm::LogWarning("Generator|Herwig7HepMC3Hadronizer")
      << "Event " << evtnum << " not generated (likely skipped in merging)";
  return false;
}

void Herwig7HepMC3Hadronizer::finalizeEvent() {
  eventInfo3() = std::make_unique<GenEventInfoProduct3>(event3().get());
  eventInfo3()->setBinningValues(std::vector<double>(1, pthat(thepegEvent)));

  if (eventsToPrint) {
    eventsToPrint--;
    //event3()->print();
    HepMC3::Print::listing(*(event3().get()));
  }

  //  if (iobc_.get())
  //    iobc_->write_event(event().get());

  edm::LogInfo("Generator|Herwig7HepMC3Hadronizer") << "Event produced";
}

bool Herwig7HepMC3Hadronizer::decay() { return true; }

bool Herwig7HepMC3Hadronizer::residualDecay() { return true; }

std::unique_ptr<GenLumiInfoHeader> Herwig7HepMC3Hadronizer::getGenLumiInfoHeader() const {
  auto genLumiInfoHeader = BaseHadronizer::getGenLumiInfoHeader();

  if (thepegEvent) {
    int weights_number = thepegEvent->optionalWeights().size();

    if (weights_number > 1) {
      genLumiInfoHeader->weightNames().reserve(weights_number + 1);
      genLumiInfoHeader->weightNames().push_back("nominal");
      std::map<std::string, double> weights_map = thepegEvent->optionalWeights();
      for (std::map<std::string, double>::iterator it = weights_map.begin(); it != weights_map.end(); it++) {
        genLumiInfoHeader->weightNames().push_back(it->first);
      }
    }
  }

  return genLumiInfoHeader;
}

void Herwig7HepMC3Hadronizer::randomizeIndex(edm::LuminosityBlock const& lumi, CLHEP::HepRandomEngine* rengine) {
  BaseHadronizer::randomizeIndex(lumi, rengine);

  if (firstLumiBlock == 0) {
    firstLumiBlock = lumi.id().luminosityBlock();
  }
  currentLumiBlock = lumi.id().luminosityBlock();
}

#include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"

typedef edm::GeneratorFilter<Herwig7HepMC3Hadronizer, gen::ExternalDecayDriver> Herwig7HepMC3GeneratorFilter;
DEFINE_FWK_MODULE(Herwig7HepMC3GeneratorFilter);

typedef edm::HadronizerFilter<Herwig7HepMC3Hadronizer, gen::ExternalDecayDriver> Herwig7HepMC3HadronizerFilter;
DEFINE_FWK_MODULE(Herwig7HepMC3HadronizerFilter);

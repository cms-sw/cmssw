#include "GeneratorInterface/ExternalDecays/interface/ConcurrentExternalDecayDriver.h"
#include "GeneratorInterface/ExternalDecays/interface/EvtGenThreadOwner.h"

#include "GeneratorInterface/Core/interface/FortranInstance.h"
#include "GeneratorInterface/EvtGenInterface/interface/EvtGenFactory.h"
#include "GeneratorInterface/EvtGenInterface/interface/EvtGenInterfaceBase.h"
#include "GeneratorInterface/TauolaInterface/interface/TauolaFactory.h"
#include "GeneratorInterface/TauolaInterface/interface/TauolaInterfaceBase.h"
#include "GeneratorInterface/PhotosInterface/interface/PhotosFactory.h"
#include "GeneratorInterface/PhotosInterface/interface/PhotosInterfaceBase.h"
#include "HepMC/GenEvent.h"
#include "FWCore/Concurrency/interface/SharedResourceNames.h"
// LHE Run
#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"
#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"

// LHE Event
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"

using namespace gen;
using namespace edm;

// Matches SimG4Core's OscarMTProducer default (`workerThreadStackSize`).
constexpr int kEvtGenThreadStackSize = 10 * 1024 * 1024;

ConcurrentExternalDecayDriver::ConcurrentExternalDecayDriver(const ParameterSet& pset)
    : fIsInitialized(false), fThreadOwner(std::make_unique<EvtGenThreadOwner>(kEvtGenThreadStackSize)) {
  std::vector<std::string> extGenNames = pset.getParameter<std::vector<std::string> >("parameterSets");

  for (unsigned int ip = 0; ip < extGenNames.size(); ++ip) {
    const std::string& curSet = extGenNames[ip];
    if (curSet == "EvtGen1" || curSet == "EvtGen130") {
      fThreadOwner->run([&]() {
        fEvtGenInterface = std::unique_ptr<EvtGenInterfaceBase>(
            EvtGenFactory::get()->create("EvtGen130", pset.getUntrackedParameter<ParameterSet>(curSet)));
      });
    } else {
      throw cms::Exception("ThreadUnsafeDecayer") << "The decayer " << curSet << " is not thread-friendly.";
    }
  }
}

ConcurrentExternalDecayDriver::~ConcurrentExternalDecayDriver() {
  if (fEvtGenInterface) {
    fThreadOwner->run([&]() { fEvtGenInterface.reset(); });
  }
}

HepMC::GenEvent* ConcurrentExternalDecayDriver::decay(HepMC::GenEvent* evt, lhef::LHEEvent* lheEvent) {
  return decay(evt);
}

HepMC::GenEvent* ConcurrentExternalDecayDriver::decay(HepMC::GenEvent* evt) {
  if (!fIsInitialized)
    return evt;

  if (fEvtGenInterface) {
    fThreadOwner->run([&]() { evt = fEvtGenInterface->decay(evt); });
    if (!evt)
      return nullptr;
  }

  return evt;
}

void ConcurrentExternalDecayDriver::init(const edm::EventSetup& es) {
  if (fIsInitialized)
    return;

  if (fEvtGenInterface) {
    fThreadOwner->run([&]() { fEvtGenInterface->init(); });
    for (std::vector<int>::const_iterator i = fEvtGenInterface->operatesOnParticles().begin();
         i != fEvtGenInterface->operatesOnParticles().end();
         i++)
      fPDGs.push_back(*i);
    for (unsigned int iss = 0; iss < fEvtGenInterface->specialSettings().size(); iss++) {
      fSpecialSettings.push_back(fEvtGenInterface->specialSettings()[iss]);
    }
  }

  fIsInitialized = true;

  return;
}

void ConcurrentExternalDecayDriver::statistics() const { return; }

void ConcurrentExternalDecayDriver::setRandomEngine(CLHEP::HepRandomEngine* v) {
  if (fEvtGenInterface) {
    fThreadOwner->run([&]() { fEvtGenInterface->setRandomEngine(v); });
  }
}

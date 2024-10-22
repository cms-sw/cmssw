/*
 *  $Date: 2010/01/19 16:17:26 $
 *  $Revision: 1.7 $
 *  \author Julia Yarba
 */

#include <ostream>
#include <memory>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include "IOMC/ParticleGuns/interface/BaseRandomtXiGunProducer.h"

#include <iostream>

using namespace edm;
using namespace std;
using namespace CLHEP;

BaseRandomtXiGunProducer::BaseRandomtXiGunProducer(const edm::ParameterSet& pset)
    : fPDGTableToken(esConsumes<Transition::BeginRun>()), fEvt(nullptr) {
  Service<RandomNumberGenerator> rng;
  if (!rng.isAvailable()) {
    throw cms::Exception("Configuration")
        << "The RandomNumberProducer module requires the RandomNumberGeneratorService\n"
           "which appears to be absent.  Please add that service to your configuration\n"
           "or remove the modules that require it.";
  }

  ParameterSet pgun_params = pset.getParameter<ParameterSet>("PGunParameters");

  // although there's the method ParameterSet::empty(),
  // it looks like it's NOT even necessary to check if it is,
  // before trying to extract parameters - if it is empty,
  // the default values seem to be taken
  fPartIDs = pgun_params.getParameter<vector<int> >("PartID");
  fMinPhi = pgun_params.getParameter<double>("MinPhi");
  fMaxPhi = pgun_params.getParameter<double>("MaxPhi");
  fECMS = pgun_params.getParameter<double>("ECMS");
  fpEnergy = fECMS / 2.0;

  fVerbosity = pset.getUntrackedParameter<int>("Verbosity", 0);

  fFireBackward = pset.getParameter<bool>("FireBackward");
  fFireForward = pset.getParameter<bool>("FireForward");

  produces<GenRunInfoProduct, Transition::EndRun>();
}

BaseRandomtXiGunProducer::~BaseRandomtXiGunProducer() {}

void BaseRandomtXiGunProducer::beginRun(const edm::Run& r, const EventSetup& es) {
  fPDGTable = es.getHandle(fPDGTableToken);
  return;
}

void BaseRandomtXiGunProducer::endRun(const Run& run, const EventSetup& es) {}
void BaseRandomtXiGunProducer::endRunProduce(Run& run, const EventSetup& es) {
  // just create an empty product
  // to keep the EventContent definitions happy
  // later on we might put the info into the run info that this is a PGun
  run.put(std::make_unique<GenRunInfoProduct>());
}

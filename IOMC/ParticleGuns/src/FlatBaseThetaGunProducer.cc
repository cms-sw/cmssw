#include <iostream>
#include <memory>

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include "IOMC/ParticleGuns/interface/FlatBaseThetaGunProducer.h"


using namespace edm;

FlatBaseThetaGunProducer::FlatBaseThetaGunProducer(const edm::ParameterSet& pset) :
   fEvt(nullptr) {

  edm::ParameterSet pgun_params = pset.getParameter<edm::ParameterSet>("PGunParameters") ;
  
  fPartIDs  = pgun_params.getParameter<std::vector<int> >("PartID");  
  fMinTheta = pgun_params.getParameter<double>("MinTheta");
  fMaxTheta = pgun_params.getParameter<double>("MaxTheta");
  fMinPhi   = pgun_params.getParameter<double>("MinPhi");
  fMaxPhi   = pgun_params.getParameter<double>("MaxPhi");
  fVerbosity = pset.getUntrackedParameter<int>( "Verbosity",0 ) ;

  fAddAntiParticle = pset.getParameter<bool>("AddAntiParticle") ;

  edm::Service<edm::RandomNumberGenerator> rng;
  if(!rng.isAvailable()) {
    throw cms::Exception("Configuration")
      << "The module inheriting from FlatBaseThetaGunProducer requires the\n"
         "RandomNumberGeneratorService, which appears to be absent.  Please\n"
         "add that service to your configuration or remove the modules that"
         "require it.";
  }

  produces<GenRunInfoProduct, Transition::EndRun>();
}

FlatBaseThetaGunProducer::~FlatBaseThetaGunProducer() {
}


void FlatBaseThetaGunProducer::beginRun(const edm::Run &r, const edm::EventSetup& es ) {
   es.getData( fPDGTable ) ;
}
void FlatBaseThetaGunProducer::endRun(const Run &run, const EventSetup& es ) {
}

void FlatBaseThetaGunProducer::endRunProduce(Run &run, const EventSetup& es )
{
   // just create an empty product
   // to keep the EventContent definitions happy
   // later on we might put the info into the run info that this is a PGun
   std::unique_ptr<GenRunInfoProduct> genRunInfo( new GenRunInfoProduct() );
   run.put(std::move(genRunInfo));
}

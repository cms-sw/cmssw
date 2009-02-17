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

namespace {

  CLHEP::HepRandomEngine& getEngineReference() {

    edm::Service<edm::RandomNumberGenerator> rng;
    if(!rng.isAvailable()) {
      throw cms::Exception("Configuration")
	<< "The RandomNumberProducer module requires the RandomNumberGeneratorService\n"
	"which appears to be absent.  Please add that service to your configuration\n"
	"or remove the modules that require it.";
    }

    // The Service has already instantiated an engine.  Make contact with it.
    return (rng->getEngine());
  }
}

FlatBaseThetaGunProducer::FlatBaseThetaGunProducer(const edm::ParameterSet& pset) :
   fEvt(0),
   fRandomEngine(getEngineReference()),
   fRandomGenerator(0) {

  edm::ParameterSet defpset ;
  edm::ParameterSet pgun_params = 
    pset.getUntrackedParameter<edm::ParameterSet>("PGunParameters", defpset ) ;
  
  std::vector<int> defids ;
  defids.push_back(13) ;
  fPartIDs  = pgun_params.getUntrackedParameter<std::vector<int> >("PartID",defids);  
  fMinTheta = pgun_params.getUntrackedParameter<double>("MinTheta",0.0);
  fMaxTheta = pgun_params.getUntrackedParameter<double>("MaxTheta",3.14159265358979323846);
  fMinPhi   = pgun_params.getUntrackedParameter<double>("MinPhi",-3.14159265358979323846);
  fMaxPhi   = pgun_params.getUntrackedParameter<double>("MaxPhi", 3.14159265358979323846);
  fVerbosity = pset.getUntrackedParameter<int>( "Verbosity",0 ) ;

  // The Service has already instantiated an engine.  Use it.
   fRandomGenerator = new RandFlat(fRandomEngine) ;
   fAddAntiParticle = pset.getUntrackedParameter("AddAntiParticle", false) ;

   produces<GenRunInfoProduct, InRun>();
}

FlatBaseThetaGunProducer::~FlatBaseThetaGunProducer() {
//  if ( fRandomGenerator != NULL ) delete fRandomGenerator;
}


void FlatBaseThetaGunProducer::beginJob( const edm::EventSetup& es ) {
   es.getData( fPDGTable ) ;
}

void FlatBaseThetaGunProducer::endRun( Run &run, const EventSetup& es )
{
   // just create an empty product
   // to keep the EventContent definitions happy
   // later on we might put the info into the run info that this is a PGun
   std::auto_ptr<GenRunInfoProduct> genRunInfo( new GenRunInfoProduct() );
   run.put( genRunInfo );
}

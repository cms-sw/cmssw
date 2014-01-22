#include "IOMC/EventVertexGenerators/interface/GaussianZBeamSpotFilter.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "HepMC/GenRanges.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Random/RandomEngine.h"

GaussianZBeamSpotFilter::GaussianZBeamSpotFilter(const edm::ParameterSet& iPSet):  
  src_(iPSet.getParameter<edm::InputTag>("src")),
  baseSZ_(iPSet.getParameter<double>("baseSZ")*cm),
  baseZ0_(iPSet.getParameter<double>("baseZ0")*cm),
  newSZ_(iPSet.getParameter<double>("newSZ")*cm),
  newZ0_(iPSet.getParameter<double>("newZ0")*cm)
{
  edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable()) {
    throw cms::Exception("Configuration")
      << "The GaussianZBeamSpotFilter requires the RandomNumberGeneratorService\n"
      "which is not present in the configuration file.  You must add the service\n"
      "in the configuration file or remove the modules that require it.";
  }

  if ( baseZ0_ != newZ0_ ) {
    edm::LogError("InconsistentZPosition") << "Z0 : old " << baseZ0_ << " and new " << newZ0_ << " are different";
  }

  if ( baseSZ_ < newSZ_ ) {
    edm::LogError("InconsistentZSigma") << "Sigma Z : old " << baseSZ_ << " smaller than new " << newSZ_ ;
  }

}

GaussianZBeamSpotFilter::~GaussianZBeamSpotFilter() {}

bool GaussianZBeamSpotFilter::filter(edm::Event& iEvent,const edm::EventSetup& iSetup)
{ 

  bool pass = true;

  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine& engine = rng->getEngine(iEvent.streamID());

  edm::Handle<edm::HepMCProduct> HepMCEvt ;
   
  iEvent.getByLabel( src_, HepMCEvt ) ;  

  //  HepMCEvt->GetEvent()->print();

  HepMC::GenEvent::vertex_const_iterator vitr= HepMCEvt->GetEvent()->vertex_range().begin();

  if ( vitr != HepMCEvt->GetEvent()->vertex_range().end() ) { 

    double vtxZ = (*vitr)->point3d().z();
    double gaussRatio = std::exp(-(vtxZ-newZ0_)*(vtxZ-newZ0_)/(2.0*newSZ_*newSZ_)+(vtxZ-baseZ0_)*(vtxZ-baseZ0_)/(2.0*baseSZ_*baseSZ_));
    if ( engine.flat() > gaussRatio ) { pass = false; }
    //    std::cout << "base sigmaZ = " << baseSZ_ << " new sigmaZ = " << newSZ_ << " vtxZ = " << vtxZ << " gaussian ratio = " << gaussRatio << " pass = " << pass << std::endl;

  }

  return pass;
  
}


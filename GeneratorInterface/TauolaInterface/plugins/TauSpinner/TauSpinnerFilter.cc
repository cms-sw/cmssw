#include "GeneratorInterface/TauolaInterface/interface/TauSpinnerFilter.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"

TauSpinnerFilter::TauSpinnerFilter(const edm::ParameterSet& pset):
  src_(pset.getParameter<edm::InputTag>("src"))
  ,fRandomEngine(nullptr)
  ,ntaus_(0)
{
  if(pset.getParameter<int>("ntaus")==1)ntaus_=1.0;
  if(pset.getParameter<int>("ntaus")==2)ntaus_=2.0;
  WTToken_=consumes<double>(src_);
}

bool TauSpinnerFilter::filter(edm::Event& e, edm::EventSetup const& es){
  edm::Service<edm::RandomNumberGenerator> rng;
  if(!rng.isAvailable()) {
    throw cms::Exception("Configuration")
      << "The RandomNumberProducer module requires the RandomNumberGeneratorService\n"
      "which appears to be absent.  Please add that service to your configuration\n"
      "or remove the modules that require it." << std::endl;
  }
  fRandomEngine = &rng->getEngine();

  edm::Handle<double> WT;
  e.getByToken(WTToken_,WT);
  if(*(WT.product())>=0 && *(WT.product())<=4.0){
    double weight=(*(WT.product()));
    if(fRandomEngine->flat()*ntaus_*2.0<weight){return true;}
  }
  return false; 
}

DEFINE_FWK_MODULE(TauSpinnerFilter);

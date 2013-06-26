#include "FastPixelCPEESProducer.h"
#include "FastPixelCPE.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"



#include <string>
#include <memory>

using namespace edm;

FastPixelCPEESProducer::FastPixelCPEESProducer(const edm::ParameterSet & p) 
{
  std::string myname = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,myname);
}

FastPixelCPEESProducer::~FastPixelCPEESProducer() {}

boost::shared_ptr<PixelClusterParameterEstimator>
FastPixelCPEESProducer::produce(const TkPixelCPERecord & iRecord){ 
  
  cpe_  = boost::shared_ptr<PixelClusterParameterEstimator>(new FastPixelCPE() );

  return cpe_;
}



#include "FastStripCPEESProducer.h"
#include "FastStripCPE.h"

#include <string>
#include <memory>

using namespace edm;

FastStripCPEESProducer::FastStripCPEESProducer(const edm::ParameterSet & p) 
{
  std::string myname = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,myname);
}

FastStripCPEESProducer::~FastStripCPEESProducer() {}

boost::shared_ptr<StripClusterParameterEstimator> 
FastStripCPEESProducer::produce(const TkStripCPERecord & iRecord){ 
  
  _cpe  = boost::shared_ptr<StripClusterParameterEstimator>(new FastStripCPE());
  
  return _cpe;
}

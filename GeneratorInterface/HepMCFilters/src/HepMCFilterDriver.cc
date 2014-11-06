#include "GeneratorInterface/HepMCFilters/interface/HepMCFilterDriver.h"
#include "GeneratorInterface/HepMCFilters/interface/GenericDauHepMCFilter.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

HepMCFilterDriver::HepMCFilterDriver(const edm::ParameterSet& pset) :
  filter_(0),
  ntried_(0),
  naccepted_(0),
  weighttried_(0.),
  weightaccepted_(0.)
{
    
  std::string filterName = pset.getParameter<std::string>("filterName");
  edm::ParameterSet filterParameters = pset.getParameter<edm::ParameterSet>("filterParameters");
  
  if (filterName=="GenericDauHepMCFilter") {
    filter_ = new GenericDauHepMCFilter(filterParameters);
  }
  else {
    edm::LogError("HepMCFilterDriver")<< "Invalid HepMCFilter name:" << filterName;
  }
  
}

HepMCFilterDriver::~HepMCFilterDriver()
{
  if (filter_) delete filter_;
  
}

bool HepMCFilterDriver::filter(const HepMC::GenEvent* evt, double weight)
{ 
  ++ntried_;
  weighttried_ += weight;
  
  bool accepted = filter_->filter(evt);
  
  if (accepted) {
    ++naccepted_;
    weightaccepted_ += weight;
  }
  
  return accepted;
}

void HepMCFilterDriver::statistics() const
{ 

  printf("ntried = %i, naccepted = %i, efficiency = %5f\n",ntried_,naccepted_,(double)naccepted_/(double)ntried_);
  printf("weighttried = %5f, weightaccepted = %5f, efficiency = %5f\n",weighttried_,weightaccepted_,weightaccepted_/weighttried_);
  
}

void HepMCFilterDriver::resetStatistics() {
 
  ntried_ = 0;
  naccepted_ = 0;
  weighttried_ = 0.;
  weightaccepted_ = 0.;
  
}
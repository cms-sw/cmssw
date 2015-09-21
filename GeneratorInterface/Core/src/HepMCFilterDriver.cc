#include "GeneratorInterface/Core/interface/HepMCFilterDriver.h"
#include "GeneratorInterface/Core/interface/GenericDauHepMCFilter.h"
#include "GeneratorInterface/Core/interface/PartonShowerBsHepMCFilter.h"


#include "FWCore/MessageLogger/interface/MessageLogger.h"

HepMCFilterDriver::HepMCFilterDriver(const edm::ParameterSet& pset) :
  filter_(0),
  numEventsPassPos_(0),
  numEventsPassNeg_(0),
  numEventsTotalPos_(0),
  numEventsTotalNeg_(0),
  sumpass_w_(0.),
  sumpass_w2_(0.),
  sumtotal_w_(0.),
  sumtotal_w2_(0.)
{
    
  std::string filterName = pset.getParameter<std::string>("filterName");
  edm::ParameterSet filterParameters = pset.getParameter<edm::ParameterSet>("filterParameters");
  
  if (filterName=="GenericDauHepMCFilter") {
    filter_ = new GenericDauHepMCFilter(filterParameters);
  }
  else if (filterName=="PartonShowerBsHepMCFilter") {
    filter_ = new PartonShowerBsHepMCFilter(filterParameters);
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
  if(weight>0)
    numEventsTotalPos_++;
  else
    numEventsTotalNeg_++;

  sumtotal_w_ += weight;
  sumtotal_w2_ += weight*weight;

  
  bool accepted = filter_->filter(evt);
  
  if (accepted) {

    if(weight>0)
      numEventsPassPos_++;
    else
      numEventsPassNeg_++;
    sumpass_w_   += weight;
    sumpass_w2_ += weight*weight;

  }
  
  return accepted;
}

void HepMCFilterDriver::statistics() const
{ 

  unsigned int ntried_    = numEventsTotalPos_ + numEventsTotalNeg_;
  unsigned int naccepted_ = numEventsPassPos_ + numEventsPassNeg_;
  printf("ntried = %i, naccepted = %i, efficiency = %5f\n",ntried_,naccepted_,(double)naccepted_/(double)ntried_);
  printf("weighttried = %5f, weightaccepted = %5f, efficiency = %5f\n",sumtotal_w_,sumpass_w_,sumpass_w_/sumtotal_w_);
  
}


void HepMCFilterDriver::resetStatistics() {
 
  numEventsPassPos_  = 0;
  numEventsPassNeg_  = 0;
  numEventsTotalPos_ = 0;
  numEventsTotalNeg_ = 0;
  sumpass_w_         = 0;
  sumpass_w2_        = 0;
  sumtotal_w_        = 0;
  sumtotal_w2_       = 0;
  
}
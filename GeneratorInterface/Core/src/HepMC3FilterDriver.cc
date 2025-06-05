#include "GeneratorInterface/Core/interface/HepMC3FilterDriver.h"
// #include "GeneratorInterface/Core/interface/GenericDauHepMC3Filter.h"
// #include "GeneratorInterface/Core/interface/PartonShowerBsHepMC3Filter.h"
// #include "GeneratorInterface/Core/interface/PartonShowerCsHepMC3Filter.h"
// #include "GeneratorInterface/Core/interface/EmbeddingHepMC3Filter.h"
// #include "GeneratorInterface/Core/interface/TaggedProtonHepMC3Filter.h"
// #include "GeneratorInterface/Core/interface/PythiaHepMC3FilterGammaGamma.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

HepMC3FilterDriver::HepMC3FilterDriver(const edm::ParameterSet& pset)
    : filter_(nullptr),
      numEventsPassPos_(0),
      numEventsPassNeg_(0),
      numEventsTotalPos_(0),
      numEventsTotalNeg_(0),
      sumpass_w_(0.),
      sumpass_w2_(0.),
      sumtotal_w_(0.),
      sumtotal_w2_(0.) {
  std::string filterName = pset.getParameter<std::string>("filterName");
  edm::ParameterSet filterParameters = pset.getParameter<edm::ParameterSet>("filterParameters");

  //   if (filterName == "GenericDauHepMC3Filter") {
  //     filter_ = new GenericDauHepMC3Filter(filterParameters);
  //   } else if (filterName == "PartonShowerBsHepMC3Filter") {
  //     filter_ = new PartonShowerBsHepMC3Filter(filterParameters);
  //   } else if (filterName == "PartonShowerCsHepMC3Filter") {
  //     filter_ = new PartonShowerCsHepMC3Filter(filterParameters);
  //   } else if (filterName == "TaggedProtonHepMC3Filter") {
  //     filter_ = new TaggedProtonHepMC3Filter(filterParameters);
  //   } else if (filterName == "EmbeddingHepMC3Filter") {
  //     filter_ = new EmbeddingHepMC3Filter(filterParameters);
  //   } else if (filterName == "PythiaHepMC3FilterGammaGamma") {
  //     filter_ = new PythiaHepMC3FilterGammaGamma(filterParameters);
  //   } else {
  throw edm::Exception(edm::errors::Configuration, "HepMC3FilterDriver") << "Invalid HepMC3Filter name:" << filterName;
  //   }
}

HepMC3FilterDriver::~HepMC3FilterDriver() {
  if (filter_)
    delete filter_;
}

bool HepMC3FilterDriver::filter(const HepMC3::GenEvent* evt, double weight) {
  if (weight > 0)
    numEventsTotalPos_++;
  else
    numEventsTotalNeg_++;

  sumtotal_w_ += weight;
  sumtotal_w2_ += weight * weight;

  bool accepted = filter_->filter(evt);

  if (accepted) {
    if (weight > 0)
      numEventsPassPos_++;
    else
      numEventsPassNeg_++;
    sumpass_w_ += weight;
    sumpass_w2_ += weight * weight;
  }

  return accepted;
}

void HepMC3FilterDriver::statistics() const {
  unsigned int ntried_ = numEventsTotalPos_ + numEventsTotalNeg_;
  unsigned int naccepted_ = numEventsPassPos_ + numEventsPassNeg_;
  printf("ntried = %i, naccepted = %i, efficiency = %5f\n", ntried_, naccepted_, (double)naccepted_ / (double)ntried_);
  printf(
      "weighttried = %5f, weightaccepted = %5f, efficiency = %5f\n", sumtotal_w_, sumpass_w_, sumpass_w_ / sumtotal_w_);
}

void HepMC3FilterDriver::resetStatistics() {
  numEventsPassPos_ = 0;
  numEventsPassNeg_ = 0;
  numEventsTotalPos_ = 0;
  numEventsTotalNeg_ = 0;
  sumpass_w_ = 0;
  sumpass_w2_ = 0;
  sumtotal_w_ = 0;
  sumtotal_w2_ = 0;
}

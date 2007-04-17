// livio.fano@cern.ch
#ifndef COSMICTIFFILTER_H
#define COSMICTIFFILTER_H


#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

namespace cms
{
class CosmicTIFFilter : public edm::EDFilter {
  public:
  CosmicTIFFilter(const edm::ParameterSet& conf);
  virtual ~CosmicTIFFilter() {}
  //   virtual bool filter(edm::Event & e, edm::EventSetup const& c);
  bool filter(edm::Event & iEvent, edm::EventSetup const& c);
  bool Sci_trig(HepLorentzVector,  HepLorentzVector, HepLorentzVector);

 private:
  edm::ParameterSet conf_;
  bool inTK;
  std::vector<double> zBounds;
  std::vector<double> rBounds;
  std::vector<double> bFields;
  double bReduction;

  };
}

#endif 

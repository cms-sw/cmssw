// patrick.janot@cern.ch, livio.fano@cern.ch
#ifndef COSMICGENFILTER_H
#define COSMICGENFILTER_H




#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

namespace cms
{
class CosmicGenFilter : public edm::EDFilter {
  public:
  CosmicGenFilter(const edm::ParameterSet& conf);
  virtual ~CosmicGenFilter() {}
  //   virtual bool filter(edm::Event & e, edm::EventSetup const& c);
  bool filter(edm::Event & iEvent, edm::EventSetup const& c);

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

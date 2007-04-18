// livio.fano@cern.ch
#ifndef COSMICTIFFILTER_H
#define COSMICTIFFILTER_H


#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "AnalysisExamples/SiStripDetectorPerformance/interface/CosmicGenFilter.h"
#include "FastSimulation/BaseParticlePropagator/interface/BaseParticlePropagator.h"

#include <map>
#include <vector>

using namespace std;
namespace cms

{
class CosmicTIFFilter : public edm::EDFilter {
  public:
  explicit CosmicTIFFilter(const edm::ParameterSet& conf);
  virtual ~CosmicTIFFilter() {}
  //   virtual bool filter(edm::Event & e, edm::EventSetup const& c);
  bool filter(edm::Event & iEvent, edm::EventSetup const& c);
  bool Sci_trig(CLHEP::HepLorentzVector,  HepLorentzVector, HepLorentzVector);

 private:
  edm::ParameterSet conf_;
  std::vector<double> Sci1;
  std::vector<double> Sci2;
  std::vector<double> Sci3;


  };
}

#endif 

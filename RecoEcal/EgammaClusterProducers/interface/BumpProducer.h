#ifndef RecoEcal_EgammaClusterProducers_BumpProducer_h
#define RecoEcal_EgammaClusterProducers_BumpProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"


// BumpProducer inherits from EDProducer, so it can be a module:
class BumpProducer : public edm::EDProducer {
  
 public:
  
  BumpProducer (const edm::ParameterSet& ps);
  
  ~BumpProducer();
  
  virtual void produce(edm::Event& evt, const edm::EventSetup& es);
  
 private:
  
  double tPhi, tEta; 
  std::vector<double> maxE;
  std::vector<int> etas, phis;

  double meanBackground;
  double backgroundFluctuation;

  std::string digiProducer_;   // name of module/plugin/producer making digis
  std::string digiCollection_; // secondary name given to collection of digis
  std::string hitCollection_;  // secondary name to be given to collection of hits
  
  int nMaxPrintout_; // max # of printouts
  int nEvt_;         // internal counter of events
  
  bool counterExceeded() const { return ((nEvt_ > nMaxPrintout_) || (nMaxPrintout_ < 0)); }

  double getBumpEnergy(double Emax, int bumpPhi, int bumpEta, int phiIndex, int etaIndex);
  double getBackgroundEnergy();

};
#endif

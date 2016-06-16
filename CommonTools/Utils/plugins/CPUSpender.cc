#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

class CPUSpender: public edm::stream::EDAnalyzer<> {

 public:
  /// Constructor
  CPUSpender(const edm::ParameterSet& pset) { timePerEvent_= pset.getUntrackedParameter<int>("secPerEvent");}

  /// Destructor
  virtual ~CPUSpender() {}

  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup) {
    time_t s = time(0);
    while (time(0)-s < timePerEvent_) { continue;}
  }

  // Operations
  void beginJob() {}
  void endJob() {}
  
protected:
  //  void printTrackRecHits(const reco::Track &, edm::ESHandle<GlobalTrackingGeometry>) const;
  
 private:
  unsigned int timePerEvent_;
};


DEFINE_FWK_MODULE(CPUSpender);


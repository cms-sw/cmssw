#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

class CPUSpender : public edm::stream::EDAnalyzer<> {
public:
  /// Constructor
  CPUSpender(const edm::ParameterSet& pset) { timePerEvent_ = pset.getUntrackedParameter<int>("secPerEvent"); }

  /// Destructor
  ~CPUSpender() override {}

  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) override {
    time_t s = time(nullptr);
    while (time(nullptr) - s < timePerEvent_) {
      continue;
    }
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

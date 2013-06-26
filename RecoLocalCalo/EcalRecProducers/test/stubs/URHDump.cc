#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"


#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"



class URHDump : public edm::EDAnalyzer {
   public:
  explicit URHDump(const edm::ParameterSet&);

private:
  virtual void beginJob(){}
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob(){}

private:

  edm::InputTag EBURecHitCollection_;
  edm::InputTag EEURecHitCollection_;
  edm::Handle<EcalUncalibratedRecHitCollection> EBURecHits_;
  edm::Handle<EcalUncalibratedRecHitCollection> EEURecHits_;

};


URHDump::URHDump(const edm::ParameterSet&iConfig) {
  EBURecHitCollection_ = iConfig.getParameter<edm::InputTag>("EBURecHitCollection");
  EEURecHitCollection_ = iConfig.getParameter<edm::InputTag>("EEURecHitCollection");
}

void URHDump::analyze(const edm::Event& ev, const edm::EventSetup&){

  ev.getByLabel(EBURecHitCollection_, EBURecHits_);
  ev.getByLabel(EEURecHitCollection_, EEURecHits_);

  for (auto const & h : (*EBURecHits_))
    std::cout << h.id() << " "
	      << h.amplitude() << " "
	      << h.jitter() << " "
	      << h.chi2() << " "
	      << h.outOfTimeEnergy() << " "
	      << h.outOfTimeChi2() << " "
	      << h.jitterErrorBits()
	      << std::endl;

}

//define this as a plug-in
DEFINE_FWK_MODULE(URHDump);

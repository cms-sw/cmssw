// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"

using namespace edm;
using namespace std;

//
// class declaration
//

class LHEGenericMassFilter : public edm::EDFilter {
public:
  explicit LHEGenericMassFilter(const edm::ParameterSet&);
  ~LHEGenericMassFilter() override;

private:
  bool filter(edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------

  edm::EDGetTokenT<LHEEventProduct> src_;
  int numRequired_;              // number of particles required to pass filter
  std::vector<int> particleID_;  // vector of particle IDs to look for
  double minMass_;
  double maxMass_;
  int totalEvents_;  // counters
  int passedEvents_;
};

LHEGenericMassFilter::LHEGenericMassFilter(const edm::ParameterSet& iConfig)
    : numRequired_(iConfig.getParameter<int>("NumRequired")),
      particleID_(iConfig.getParameter<std::vector<int> >("ParticleID")),
      minMass_(iConfig.getParameter<double>("MinMass")),
      maxMass_(iConfig.getParameter<double>("MaxMass")),
      totalEvents_(0),
      passedEvents_(0) {
  //here do whatever other initialization is needed
  src_ = consumes<LHEEventProduct>(iConfig.getParameter<edm::InputTag>("src"));
}

LHEGenericMassFilter::~LHEGenericMassFilter() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
}

// ------------ method called to skim the data  ------------
bool LHEGenericMassFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<LHEEventProduct> EvtHandle;
  iEvent.getByToken(src_, EvtHandle);

  totalEvents_++;
  int nFound = 0;

  double Px = 0.;
  double Py = 0.;
  double Pz = 0.;
  double E = 0.;

  for (int i = 0; i < EvtHandle->hepeup().NUP; ++i) {
    if (EvtHandle->hepeup().ISTUP[i] != 1) {  // keep only outgoing particles
      continue;
    }
    for (unsigned int j = 0; j < particleID_.size(); ++j) {
      if (abs(particleID_[j]) == abs(EvtHandle->hepeup().IDUP[i])) {
        nFound++;
        Px = Px + EvtHandle->hepeup().PUP[i][0];
        Py = Py + EvtHandle->hepeup().PUP[i][1];
        Pz = Pz + EvtHandle->hepeup().PUP[i][2];
        E = E + EvtHandle->hepeup().PUP[i][3];

        break;  // only match a given particle once!
      }
    }  // loop over targets

  }  // loop over particles

  double Mass = std::sqrt(E * E - (Px * Px + Py * Py + Pz * Pz));

  // event accept/reject logic
  if (Mass > minMass_ && Mass < maxMass_ && nFound == numRequired_) {
    passedEvents_++;
    return true;
  } else {
    return false;
  }
}

// ------------ method called once each job just after ending the event loop  ------------
void LHEGenericMassFilter::endJob() {
  edm::LogInfo("LHEGenericMassFilter") << "=== Results of LHEGenericMassFilter: passed " << passedEvents_ << "/"
                                       << totalEvents_ << " events" << std::endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(LHEGenericMassFilter);

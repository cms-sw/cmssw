// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"

using namespace edm;
using namespace std;

//
// class declaration
//

class LHEGenericMassFilter : public edm::global::EDFilter<> {
public:
  explicit LHEGenericMassFilter(const edm::ParameterSet&);
  ~LHEGenericMassFilter() override;

private:
  bool filter(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;
  void endJob() override;

  // ----------member data ---------------------------

  const edm::EDGetTokenT<LHEEventProduct> src_;
  const int numRequired_;              // number of particles required to pass filter
  const std::vector<int> particleID_;  // vector of particle IDs to look for
  const double minMass_;
  const double maxMass_;
};

LHEGenericMassFilter::LHEGenericMassFilter(const edm::ParameterSet& iConfig)
    : src_(consumes<LHEEventProduct>(iConfig.getParameter<edm::InputTag>("src"))),
      numRequired_(iConfig.getParameter<int>("NumRequired")),
      particleID_(iConfig.getParameter<std::vector<int> >("ParticleID")),
      minMass_(iConfig.getParameter<double>("MinMass")),
      maxMass_(iConfig.getParameter<double>("MaxMass")) {}

LHEGenericMassFilter::~LHEGenericMassFilter() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
}

// ------------ method called to skim the data  ------------
bool LHEGenericMassFilter::filter(edm::StreamID iID, edm::Event& iEvent, edm::EventSetup const& iSetup) const {
  edm::Handle<LHEEventProduct> EvtHandle;
  iEvent.getByToken(src_, EvtHandle);

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

  // event accept/reject logic
  if (nFound == numRequired_) {
    double Mass = std::sqrt(E * E - (Px * Px + Py * Py + Pz * Pz));
    if (Mass > minMass_ && Mass < maxMass_) {
      return true;
    }
  }
  return false;
}

// ------------ method called once each job just after ending the event loop  ------------
void LHEGenericMassFilter::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(LHEGenericMassFilter);

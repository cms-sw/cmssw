#include "GeneratorInterface/GenFilters/plugins/LHEVpTFilter.h"

using namespace edm;
using namespace std;

LHEVpTFilter::LHEVpTFilter(const edm::ParameterSet& iConfig)
    : vptMin_(iConfig.getParameter<double>("VpTMin")),
      vptMax_(iConfig.getParameter<double>("VpTMax")),
      totalEvents_(0),
      passedEvents_(0) {
  //here do whatever other initialization is needed
  src_ = consumes<LHEEventProduct>(iConfig.getParameter<edm::InputTag>("src"));
}

LHEVpTFilter::~LHEVpTFilter() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
}

// ------------ method called to skim the data  ------------
bool LHEVpTFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  lepCands.clear();
  edm::Handle<LHEEventProduct> EvtHandle;
  iEvent.getByToken(src_, EvtHandle);

  totalEvents_++;

  lheParticles = EvtHandle->hepeup().PUP;

  for (unsigned int i = 0; i < lheParticles.size(); ++i) {
    if (EvtHandle->hepeup().ISTUP[i] != 1) {  // keep only outgoing particles
      continue;
    }
    unsigned absPdgId = std::abs(EvtHandle->hepeup().IDUP[i]);
    if (absPdgId >= 11 && absPdgId <= 16) {
      lepCands.push_back(
          ROOT::Math::PxPyPzEVector(lheParticles[i][0], lheParticles[i][1], lheParticles[i][2], lheParticles[i][3]));
    }
  }
  double vpt_ = -1;
  if (lepCands.size() == 2) {
    vpt_ = (lepCands[0] + lepCands[1]).pt();
  }
  if (vpt_ <= vptMax_ && vpt_ > vptMin_) {
    passedEvents_++;
    return true;
  } else {
    return false;
  }
}

// ------------ method called once each job just after ending the event loop  ------------
void LHEVpTFilter::endJob() {
  edm::LogInfo("LHEVpTFilter") << "=== Results of LHEVpTFilter: passed " << passedEvents_ << "/" << totalEvents_
                               << " events" << std::endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(LHEVpTFilter);

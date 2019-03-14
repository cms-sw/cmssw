#include "GeneratorInterface/GenFilters/interface/LHEPtFilter.h"

using namespace edm;
using namespace std;

LHEPtFilter::LHEPtFilter(const edm::ParameterSet& iConfig) :
pdgIdVec_(iConfig.getParameter<std::vector<int>>("selectedPdgIds")),
ptMin_(iConfig.getParameter<double>("ptMin")),
ptMax_(iConfig.getParameter<double>("ptMax")),
totalEvents_(0), passedEvents_(0)
{
  //here do whatever other initialization is needed
  src_ = consumes<LHEEventProduct>(iConfig.getParameter<edm::InputTag>("src"));
  pdgIds_ = std::set<int>(pdgIdVec_.begin(), pdgIdVec_.end());
}

LHEPtFilter::~LHEPtFilter()
{

  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)

}


// ------------ method called to skim the data  ------------
bool LHEPtFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  cands.clear();
  edm::Handle< LHEEventProduct > EvtHandle ;
  iEvent.getByToken( src_ , EvtHandle ) ;
  
  totalEvents_++;

  lheParticles = EvtHandle->hepeup().PUP;

  for (unsigned int i = 0; i < lheParticles.size(); ++i) {
    if (EvtHandle->hepeup().ISTUP[i] != 1) { // keep only outgoing particles
      continue;
    }
    int pdgId = EvtHandle->hepeup().IDUP[i];
    if (pdgIds_.count(pdgId)) {
      cands.push_back(ROOT::Math::PxPyPzEVector(lheParticles[i][0],lheParticles[i][1],lheParticles[i][2],lheParticles[i][3]));
    }
  }
  double vpt_ = -1;
  if (cands.size() >= 1) {
    ROOT::Math::PxPyPzEVector tot = cands.at(0);
    for (unsigned icand = 1; icand < cands.size(); ++icand) {
      tot += cands.at(icand);
    }
    vpt_ = tot.pt();
  }
  if ((ptMax_ < 0. || vpt_ <= ptMax_) && vpt_ > ptMin_) {
    passedEvents_++;
    return true;
  } else {
    return false;
  }
  
}

// ------------ method called once each job just after ending the event loop  ------------
void LHEPtFilter::endJob() {
  edm::LogInfo("LHEPtFilter") << "=== Results of LHEPtFilter: passed "
  << passedEvents_ << "/" << totalEvents_ << " events" << std::endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(LHEPtFilter);


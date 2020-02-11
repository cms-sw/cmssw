#include "GeneratorInterface/GenFilters/plugins/LHEGenericFilter.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <string>

LHEGenericFilter::LHEGenericFilter(const edm::ParameterSet& iConfig)
    : src_(consumes<LHEEventProduct>(iConfig.getParameter<edm::InputTag>("src"))),
      numRequired_(iConfig.getParameter<int>("NumRequired")),
      particleID_(iConfig.getParameter<std::vector<int> >("ParticleID")) {
  // LT  meaning <
  // GT          >
  // EQ          =
  // NE          !=
  std::string acceptLogic = iConfig.getParameter<std::string>("AcceptLogic");
  if (acceptLogic == "LT")
    whichlogic = LT;
  else if (acceptLogic == "GT")
    whichlogic = GT;
  else if (acceptLogic == "EQ")
    whichlogic = EQ;
  else if (acceptLogic == "NE")
    whichlogic = NE;
  else {
    edm::LogError("cat_A") << "wrong input for AcceptLogic string";
    // at least initialize it to something reproducible
    whichlogic = LT;
  }
}

// ------------ method called to skim the data  ------------
bool LHEGenericFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  edm::Handle<LHEEventProduct> EvtHandle;
  iEvent.getByToken(src_, EvtHandle);

  int nFound = 0;

  for (int i = 0; i < EvtHandle->hepeup().NUP; ++i) {
    if (EvtHandle->hepeup().ISTUP[i] != 1) {  // keep only outgoing particles
      continue;
    }
    for (unsigned int j = 0; j < particleID_.size(); ++j) {
      if (particleID_[j] == 0 || abs(particleID_[j]) == abs(EvtHandle->hepeup().IDUP[i])) {
        nFound++;
        break;  // only match a given particle once!
      }
    }  // loop over targets

  }  // loop over particles

  // event accept/reject logic
  if ((whichlogic == LT && nFound < numRequired_) || (whichlogic == GT && nFound > numRequired_) ||
      (whichlogic == EQ && nFound == numRequired_) || (whichlogic == NE && nFound != numRequired_)) {
    return true;
  } else {
    return false;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(LHEGenericFilter);

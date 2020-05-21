#include "RecoMuon/TrackerSeedGenerator/plugins/CompositeTSG.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGeneratorFactory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

CompositeTSG::CompositeTSG(const edm::ParameterSet& par, edm::ConsumesCollector& IC) {
  theCategory = "CompositeTSG";

  //configure the individual components of the TSG
  std::vector<std::string> PSetNames = par.getParameter<std::vector<std::string> >("PSetNames");

  for (auto& PSetName : PSetNames) {
    edm::ParameterSet TSGpset = par.getParameter<edm::ParameterSet>(PSetName);
    if (TSGpset.empty()) {
      theNames.push_back(PSetName + ":" + "NULL");
      theTSGs.emplace_back(nullptr);
    } else {
      std::string SeedGenName = TSGpset.getParameter<std::string>("ComponentName");
      theNames.push_back(PSetName + ":" + SeedGenName);
      theTSGs.emplace_back(TrackerSeedGeneratorFactory::get()->create(SeedGenName, TSGpset, IC));
    }
  }
}

CompositeTSG::~CompositeTSG() = default;

void CompositeTSG::init(const MuonServiceProxy* service) {
  theProxyService = service;
  for (unsigned int iTSG = 0; iTSG != theTSGs.size(); iTSG++) {
    if (theTSGs[iTSG])
      theTSGs[iTSG]->init(service);
  }
}

void CompositeTSG::setEvent(const edm::Event& event) {
  theEvent = &event;
  for (unsigned int iTSG = 0; iTSG != theTSGs.size(); iTSG++) {
    if (theTSGs[iTSG])
      theTSGs[iTSG]->setEvent(event);
  }
}

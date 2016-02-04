#include "RecoMuon/TrackerSeedGenerator/plugins/CompositeTSG.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGeneratorFactory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

CompositeTSG::CompositeTSG(const edm::ParameterSet & par){
  theCategory = "CompositeTSG";

  //configure the individual components of the TSG
  std::vector<std::string> PSetNames =  par.getParameter<std::vector<std::string> >("PSetNames");

  for (std::vector<std::string>::iterator nIt = PSetNames.begin();nIt!=PSetNames.end();nIt++){
    edm::ParameterSet TSGpset = par.getParameter<edm::ParameterSet>(*nIt);
    if (TSGpset.empty()) {
      theNames.push_back((*nIt)+":"+"NULL");
      theTSGs.push_back((TrackerSeedGenerator*)(0));
    }else {
      std::string SeedGenName = TSGpset.getParameter<std::string>("ComponentName");
      theNames.push_back((*nIt)+":"+SeedGenName);
      theTSGs.push_back(TrackerSeedGeneratorFactory::get()->create(SeedGenName,TSGpset));
    }
  }
  
}

CompositeTSG::~CompositeTSG(){
  //delete the components ?
}


void CompositeTSG::init(const MuonServiceProxy* service){
  theProxyService = service;
  for (unsigned int iTSG=0; iTSG!=theTSGs.size();iTSG++){
    if(theTSGs[iTSG]) theTSGs[iTSG]->init(service);
  }
}

void CompositeTSG::setEvent(const edm::Event &event){
  theEvent = &event;
  for (unsigned int iTSG=0; iTSG!=theTSGs.size();iTSG++){
    if(theTSGs[iTSG]) theTSGs[iTSG]->setEvent(event);
  }
}

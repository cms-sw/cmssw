#include "RecoMuon/TrackerSeedGenerator/plugins/CompositeTSG.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGeneratorFactory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

#include <TH1.h>

CompositeTSG::CompositeTSG(const edm::ParameterSet & par){
  theCategory = "CompositeTSG";

  useTFileService_ = par.getUntrackedParameter<bool>("UseTFileService",false);
  edm::Service<TFileService> fs;

  //configure the individual components of the TSG
  std::vector<std::string> PSetNames =  par.getParameter<std::vector<std::string> >("PSetNames");

  for (std::vector<std::string>::iterator nIt = PSetNames.begin();nIt!=PSetNames.end();nIt++){
    edm::ParameterSet TSGpset = par.getParameter<edm::ParameterSet>(*nIt);
    if (TSGpset.empty()) {
      theNames.push_back((*nIt)+":"+"NULL");
      theTSGs.push_back((TrackerSeedGenerator*)(0));
      theHistos.push_back((TH1F*)(0));
    }else {
      std::string SeedGenName = TSGpset.getParameter<std::string>("ComponentName");
      theNames.push_back((*nIt)+":"+SeedGenName);
      TSGpset.addUntrackedParameter<bool>("UseTFileService",useTFileService_);
      theTSGs.push_back(TrackerSeedGeneratorFactory::get()->create(SeedGenName,TSGpset));
      std::string hName = "nSeedPerTrack_"+(*nIt)+"_"+SeedGenName;
      if(useTFileService_) theHistos.push_back(fs->make<TH1F>(hName.c_str(),hName.c_str(),76,-0.5,75.5));
      else theHistos.push_back((TH1F*)(0));
    }
  }
  
}

CompositeTSG::~CompositeTSG(){
  //delete the components ?
}


void CompositeTSG::init(const MuonServiceProxy* service){
  theProxyService = service;
  for (uint iTSG=0; iTSG!=theTSGs.size();iTSG++){
    if(theTSGs[iTSG]) theTSGs[iTSG]->init(service);
  }
}

void CompositeTSG::setEvent(const edm::Event &event){
  for (uint iTSG=0; iTSG!=theTSGs.size();iTSG++){
    if(theTSGs[iTSG]) theTSGs[iTSG]->setEvent(event);
  }
}

// Original Author:  Anna Cimmino

#include "DQM/RPCMonitorClient/interface/RPCRecHitProbabilityClient.h"
//Framework
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//DQMServices
#include "DQMServices/Core/interface/MonitorElement.h"

#include <string>

RPCRecHitProbabilityClient::RPCRecHitProbabilityClient(const edm::ParameterSet& iConfig){

 edm::LogVerbatim ("rpcdqmclient") << "[RPCRecHitProbabilityClient]: Constructor";

  
  std::string subsystemFolder= iConfig.getUntrackedParameter<std::string>("RPCFolder", "RPC");
  std::string recHitTypeFolder= iConfig.getUntrackedParameter<std::string>("MuonFolder", "Muon");

  std::string summaryFolder = iConfig.getUntrackedParameter<std::string>("GlobalFolder", "SummaryHistograms");

  globalFolder_ = subsystemFolder + "/"+ recHitTypeFolder + "/"+ summaryFolder ;

}

RPCRecHitProbabilityClient::~RPCRecHitProbabilityClient(){}

void RPCRecHitProbabilityClient::beginJob(){

  edm::LogVerbatim ("rpcrechitprobabilityclient") << "[RPCRecHitProbabilityClient]: Begin Job";

}


void RPCRecHitProbabilityClient::dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, edm::LuminosityBlock const &, edm::EventSetup const&){}



void  RPCRecHitProbabilityClient::dqmEndJob(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter) {
  
  edm::LogVerbatim ("rpcrechitprobabilityclient") << "[RPCRecHitProbabilityClient]: End Run";
  
  MonitorElement *  NumberOfMuonEta = igetter.get( globalFolder_ +"/NumberOfMuonEta");
  MonitorElement *  NumberOfMuonPt_B = igetter.get( globalFolder_ + "/NumberOfMuonPt_Barrel");
  MonitorElement *  NumberOfMuonPt_EP = igetter.get( globalFolder_ + "/NumberOfMuonPt_EndcapP");
  MonitorElement *  NumberOfMuonPt_EM = igetter.get( globalFolder_ + "/NumberOfMuonPt_EndcapM");
  MonitorElement *  NumberOfMuonPhi_B = igetter.get( globalFolder_ + "/NumberOfMuonPhi_Barrel");
  MonitorElement *  NumberOfMuonPhi_EP = igetter.get( globalFolder_ + "/NumberOfMuonPhi_EndcapP");
  MonitorElement *  NumberOfMuonPhi_EM = igetter.get( globalFolder_ + "/NumberOfMuonPhi_EndcapM");
  
  if(NumberOfMuonEta == 0  || 
     NumberOfMuonPt_B == 0  || NumberOfMuonPt_EP == 0  || NumberOfMuonPt_EM == 0  || 
     NumberOfMuonPhi_B == 0  || NumberOfMuonPhi_EP == 0  || NumberOfMuonPhi_EM == 0 ) return;


  TH1F *    NumberOfMuonEtaTH1F = NumberOfMuonEta->getTH1F(); 
  TH1F *    NumberOfMuonPtBTH1F = NumberOfMuonPt_B->getTH1F(); 
  TH1F *    NumberOfMuonPtEPTH1F = NumberOfMuonPt_EP->getTH1F(); 
  TH1F *    NumberOfMuonPtEMTH1F = NumberOfMuonPt_EM->getTH1F(); 
  TH1F *    NumberOfMuonPhiBTH1F = NumberOfMuonPhi_B->getTH1F(); 
  TH1F *    NumberOfMuonPhiEPTH1F = NumberOfMuonPhi_EP->getTH1F(); 
  TH1F *    NumberOfMuonPhiEMTH1F = NumberOfMuonPhi_EM->getTH1F(); 
      
  MonitorElement *  recHit;
  TH1F *  recHitTH1F; 
  std::stringstream name;
      
  for(int i = 1 ; i<= 6  ; i++) {
    
    recHit = NULL;
    recHitTH1F = NULL;

    name.str("");
    name<< globalFolder_ <<"/"<<i<<"RecHitMuonEta";
    recHit = igetter.get(name.str());

    if(recHit){
      
      recHitTH1F = recHit->getTH1F(); 
      recHitTH1F->Divide(NumberOfMuonEtaTH1F);
    }

    recHit = NULL;
    recHitTH1F = NULL;

    name.str("");
    name<< globalFolder_ <<"/"<<i<<"RecHitMuonPtB";
    recHit = igetter.get(name.str());

    if(recHit){      
      recHitTH1F = recHit->getTH1F(); 
      recHitTH1F->Divide(NumberOfMuonPtBTH1F);
    }

    recHit = NULL;
    recHitTH1F = NULL;
    
    name.str("");
    name<< globalFolder_ <<"/"<<i<<"RecHitMuonPhiB";
    recHit = igetter.get(name.str());

    if(recHit){      
      recHitTH1F = recHit->getTH1F(); 
      recHitTH1F->Divide(NumberOfMuonPhiBTH1F);
    }

    recHit = NULL;
    recHitTH1F = NULL;

    name.str("");
    name<< globalFolder_ <<"/"<<i<<"RecHitMuonPtEP";
    recHit = igetter.get(name.str());

    if(recHit){      
      recHitTH1F = recHit->getTH1F(); 
      recHitTH1F->Divide(NumberOfMuonPtEPTH1F);
    }

    recHit = NULL;
    recHitTH1F = NULL;

    name.str("");
    name<< globalFolder_ <<"/"<<i<<"RecHitMuonPhiEP";
    recHit = igetter.get(name.str());

    if(recHit){      
      recHitTH1F = recHit->getTH1F(); 
      recHitTH1F->Divide(NumberOfMuonPhiEPTH1F);
    }

    
    recHit = NULL;
    recHitTH1F = NULL;

    name.str("");
    name<< globalFolder_ <<"/"<<i<<"RecHitMuonPtEM";
    recHit = igetter.get(name.str());

    if(recHit){      
      recHitTH1F = recHit->getTH1F(); 
      recHitTH1F->Divide(NumberOfMuonPtEMTH1F);
    }

    recHit = NULL;
    recHitTH1F = NULL;

    name.str("");
    name<< globalFolder_ <<"/"<<i<<"RecHitMuonPhiEM";
    recHit = igetter.get(name.str());

    if(recHit){     
      recHitTH1F = recHit->getTH1F(); 
      recHitTH1F->Divide(NumberOfMuonPhiEMTH1F);
    }



  }

}





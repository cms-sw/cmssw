
/*
 *  See header file for a description of this class.
 *
 *  \author G. Mila - INFN Torino
 */


#include <DQMOffline/Muon/interface/MuonRecoTest.h>

// Framework
#include <FWCore/Framework/interface/Event.h>
#include "DataFormats/Common/interface/Handle.h" 
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>


#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Run.h"

#include <iostream>
#include <stdio.h>
#include <string>
#include <math.h>
#include "TF1.h"

using namespace edm;
using namespace std;


MuonRecoTest::MuonRecoTest(const edm::ParameterSet& ps){

  parameters = ps;

  prescaleFactor = parameters.getUntrackedParameter<int>("diagnosticPrescale", 1);

  // Parameters
  etaBin = parameters.getParameter<int>("etaBin");
  etaMin = parameters.getParameter<double>("etaMin");
  etaMax = parameters.getParameter<double>("etaMax");

  phiBin = parameters.getParameter<int>("phiBin");
  phiMin = parameters.getParameter<double>("phiMin");
  phiMax = parameters.getParameter<double>("phiMax");

  EfficiencyCriterionName = parameters.getUntrackedParameter<string>("efficiencyTestName","EfficiencyInRange"); 
}
void MuonRecoTest::dqmEndJob(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter) {
  
  /// BOOKING NEW HISTOGRAMS
  ibooker.setCurrentFolder("Muons/Tests/muonRecoTest");
  // efficiency plot
  etaEfficiency = ibooker.book1D("etaEfficiency_staMuon", "#eta_{STA} efficiency", etaBin, etaMin, etaMax);
  phiEfficiency = ibooker.book1D("phiEfficiency_staMuon", "#phi_{STA} efficiency", phiBin, phiMin, phiMax);

  // alignment plots
  globalRotation.push_back(ibooker.book1D("muVStkSytemRotation_posMu_profile", "pT_{TK} / pT_{GLB} vs pT_{GLB} profile for #mu^{+}",50,0,200));
  globalRotation.push_back(ibooker.book1D("muVStkSytemRotation_negMu_profile", "pT_{TK} / pT_{GLB} vs pT_{GLB} profile for #mu^{-}",50,0,200));
  globalRotation.push_back(ibooker.book1D("muVStkSytemRotation_profile", "pT_{TK} / pT_{GLB} vs pT_{GLB} profile for #mu^{+}-#mu^{-}",50,0,200));


  /// GETTING PREVIOUS HISTOS AND DO SOME OPERATIONS
  string path = "Muons/MuonRecoAnalyzer/StaEta_ifCombinedAlso";
  MonitorElement * staEtaIfComb_histo = igetter.get(path);
  path = "Muons/MuonRecoAnalyzer/StaEta";
  MonitorElement * staEta_histo = igetter.get(path);

  if(staEtaIfComb_histo && staEta_histo){
    TH1F * staEtaIfComb_root = staEtaIfComb_histo->getTH1F();
    TH1F * staEta_root = staEta_histo->getTH1F();

    if(staEtaIfComb_root->GetXaxis()->GetNbins()!=etaBin
       || staEtaIfComb_root->GetXaxis()->GetXmax()!=etaMax
       || staEtaIfComb_root->GetXaxis()->GetXmin()!=etaMin){
      LogTrace(metname)<<"[MuonRecoTest] wrong histo binning on eta histograms";
      return;
    }

    for(int i=1; i<=etaBin; i++){
      if(staEta_root->GetBinContent(i)!=0){
	double efficiency = double(staEtaIfComb_root->GetBinContent(i))/double(staEta_root->GetBinContent(i));
	etaEfficiency->setBinContent(i,efficiency);
      }
    }
  }

  path = "Muons/MuonRecoAnalyzer/StaPhi_ifCombinedAlso";
  MonitorElement * staPhiIfComb_histo = igetter.get(path);
  path = "Muons/MuonRecoAnalyzer/StaPhi";
  MonitorElement * staPhi_histo = igetter.get(path);

  if(staPhiIfComb_histo && staPhi_histo){
 
    TH1F * staPhiIfComb_root = staPhiIfComb_histo->getTH1F();
    TH1F * staPhi_root = staPhi_histo->getTH1F();

    if(staPhiIfComb_root->GetXaxis()->GetNbins()!=phiBin
       || staPhiIfComb_root->GetXaxis()->GetXmax()!=phiMax
       || staPhiIfComb_root->GetXaxis()->GetXmin()!=phiMin){
      LogTrace(metname)<<"[MuonRecoTest] wrong histo binning on phi histograms";
      return;
    }

    for(int i=1; i<=etaBin; i++){
      if(staPhi_root->GetBinContent(i)!=0){
	double efficiency = double(staPhiIfComb_root->GetBinContent(i))/double(staPhi_root->GetBinContent(i));
	phiEfficiency->setBinContent(i,efficiency);
      }
    }
  }


  // efficiency test 
  

  // eta efficiency
  const QReport * theEtaQReport = etaEfficiency->getQReport(EfficiencyCriterionName);
  if(theEtaQReport) {
    vector<dqm::me_util::Channel> badChannels = theEtaQReport->getBadChannels();
    for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	 channel != badChannels.end(); channel++) {
      LogTrace(metname)<<"[etaEfficiency test] bad ranges: "<<(*channel).getBin()<<"  Contents : "<<(*channel).getContents()<<endl;
    }
    LogTrace(metname)<< "-------- type: [etaEfficiency]  "<<theEtaQReport->getMessage()<<" ------- "<<theEtaQReport->getStatus()<<endl;
  }
  // phi efficiency
  const QReport * thePhiQReport = phiEfficiency->getQReport(EfficiencyCriterionName);
  if(thePhiQReport) {
    vector<dqm::me_util::Channel> badChannels = thePhiQReport->getBadChannels();
    for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	 channel != badChannels.end(); channel++) {
      LogTrace(metname)<< "[phiEfficiency test] bad ranges: "<<(*channel).getBin()<<"  Contents : "<<(*channel).getContents()<<endl;
    }
    LogTrace(metname)<<"-------- type: [phiEfficiency]  "<<thePhiQReport->getMessage()<<" ------- "<<thePhiQReport->getStatus()<<endl;
  }

  //alignment plot
  string pathPos = "Muons/MuonRecoAnalyzer/muVStkSytemRotation_posMu";
  MonitorElement * muVStkSytemRotation_posMu_histo = igetter.get(pathPos);
  string pathNeg = "Muons/MuonRecoAnalyzer/muVStkSytemRotation_negMu";
  MonitorElement * muVStkSytemRotation_negMu_histo = igetter.get(pathNeg);
  if(muVStkSytemRotation_posMu_histo && muVStkSytemRotation_negMu_histo){

    TH2F * muVStkSytemRotation_posMu_root = muVStkSytemRotation_posMu_histo->getTH2F();
    TProfile * muVStkSytemRotation_posMu_profile = muVStkSytemRotation_posMu_root->ProfileX("",1,100);
    TH2F * muVStkSytemRotation_negMu_root = muVStkSytemRotation_negMu_histo->getTH2F();
    TProfile * muVStkSytemRotation_negMu_profile = muVStkSytemRotation_negMu_root->ProfileX("",1,100);

    for(int x=1; x<50; x++){
      globalRotation[0]->Fill((x*4)-1,muVStkSytemRotation_posMu_profile->GetBinContent(x));
      globalRotation[0]->setBinError(x,muVStkSytemRotation_posMu_profile->GetBinError(x));
      globalRotation[1]->Fill((x*4)-1,muVStkSytemRotation_negMu_profile->GetBinContent(x));
      globalRotation[1]->setBinError(x,muVStkSytemRotation_negMu_profile->GetBinError(x));
      globalRotation[2]->Fill((x*4)-1,muVStkSytemRotation_posMu_profile->GetBinContent(x)-muVStkSytemRotation_negMu_profile->GetBinContent(x));
      globalRotation[2]->setBinError(x,sqrt((muVStkSytemRotation_posMu_profile->GetBinError(x)*muVStkSytemRotation_posMu_profile->GetBinError(x))
					    + (muVStkSytemRotation_negMu_profile->GetBinError(x)*muVStkSytemRotation_negMu_profile->GetBinError(x))));
    }
  }

}
  

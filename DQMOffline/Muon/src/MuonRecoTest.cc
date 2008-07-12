
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/06/09 13:41:20 $
 *  $Revision: 1.5 $
 *  \author G. Mila - INFN Torino
 */


#include <DQMOffline/Muon/src/MuonRecoTest.h>

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

#include <iostream>
#include <stdio.h>
#include <string>
#include <math.h>
#include "TF1.h"

using namespace edm;
using namespace std;


MuonRecoTest::MuonRecoTest(const edm::ParameterSet& ps){

  cout << "[MuonRecoTest]: Constructor called!"<<endl;
  parameters = ps;

  dbe = edm::Service<DQMStore>().operator->();
  dbe->setVerbose(1);

  prescaleFactor = parameters.getUntrackedParameter<int>("diagnosticPrescale", 1);

}


MuonRecoTest::~MuonRecoTest(){

  LogTrace(metname) << "DTResolutionTest: analyzed " << nevents << " events";

}


void MuonRecoTest::beginJob(const edm::EventSetup& context){

  metname = "muonRecoTest";
  dbe->setCurrentFolder("Muons/Tests/muonRecoTest");

  LogTrace(metname)<<"[MuonRecoTest] Parameters initialization";
 
  // efficiency plot

  etaBin = parameters.getParameter<int>("etaBin");
  etaMin = parameters.getParameter<double>("etaMin");
  etaMax = parameters.getParameter<double>("etaMax");
  etaEfficiency = dbe->book1D("etaEfficiency_staMuon", "etaEfficiency_staMuon", etaBin, etaMin, etaMax);

  phiBin = parameters.getParameter<int>("phiBin");
  phiMin = parameters.getParameter<double>("phiMin");
  phiMax = parameters.getParameter<double>("phiMax");
  phiEfficiency = dbe->book1D("phiEfficiency_staMuon", "phiEfficiency_staMuon", phiBin, phiMin, phiMax);

}


void MuonRecoTest::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {

  cout<<"[MuonRecoTest]: Begin of LS transition"<<endl;

  // Get the run number
  run = lumiSeg.run();

}


void MuonRecoTest::analyze(const edm::Event& e, const edm::EventSetup& context){

  nevents++;
  LogTrace(metname)<< "[MuonRecoTest]: "<<nevents<<" events";

}



void MuonRecoTest::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {

  LogTrace(metname)<<"[MuonRecoTest]: End of LS transition, performing the DQM client operation";
  
  // counts number of lumiSegs 
  nLumiSegs = lumiSeg.id().luminosityBlock();
  
  // prescale factor
  if ( nLumiSegs%prescaleFactor != 0 ) return;

  string path = "Muons/MuonRecoAnalyzer/StaEta_ifCombinedAlso";
  MonitorElement * staEtaIfComb_histo = dbe->get(path);
  path = "Muons/MuonRecoAnalyzer/StaEta";
  MonitorElement * staEta_histo = dbe->get(path);

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
  MonitorElement * staPhiIfComb_histo = dbe->get(path);
  path = "Muons/MuonRecoAnalyzer/StaPhi";
  MonitorElement * staPhi_histo = dbe->get(path);

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
  string EfficiencyCriterionName = parameters.getUntrackedParameter<string>("efficiencyTestName","EfficiencyInRange"); 

  // eta efficiency
  const QReport * theEtaQReport = etaEfficiency->getQReport(EfficiencyCriterionName);
  if(theEtaQReport) {
    vector<dqm::me_util::Channel> badChannels = theEtaQReport->getBadChannels();
    for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	 channel != badChannels.end(); channel++) {
      cout<<"[etaEfficiency test] bad ranges: "<<(*channel).getBin()<<"  Contents : "<<(*channel).getContents()<<endl;
    }
    cout<< "-------- type: [etaEfficiency]  "<<theEtaQReport->getMessage()<<" ------- "<<theEtaQReport->getStatus()<<endl;
  }
  // phi efficiency
  const QReport * thePhiQReport = phiEfficiency->getQReport(EfficiencyCriterionName);
  if(thePhiQReport) {
    vector<dqm::me_util::Channel> badChannels = thePhiQReport->getBadChannels();
    for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	 channel != badChannels.end(); channel++) {
      cout<< "[phiEfficiency test] bad ranges: "<<(*channel).getBin()<<"  Contents : "<<(*channel).getContents()<<endl;
    }
    cout<<"-------- type: [phiEfficiency]  "<<thePhiQReport->getMessage()<<" ------- "<<thePhiQReport->getStatus()<<endl;
  }

}


void MuonRecoTest::endJob(){
  
  LogTrace(metname)<< "[MuonRecoTest] endjob called!";
  dbe->rmdir("Muons/Tests/muonRecoTest");
  
}
  

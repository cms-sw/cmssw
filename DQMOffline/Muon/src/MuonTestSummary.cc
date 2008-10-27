
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/10/22 09:38:05 $
 *  $Revision: 1.1 $
 *  \author G. Mila - INFN Torino
 */


#include <DQMOffline/Muon/src/MuonTestSummary.h>

// Framework
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>

using namespace edm;
using namespace std;


MuonTestSummary::MuonTestSummary(const edm::ParameterSet& ps){

  dbe = Service<DQMStore>().operator->();

  // parameter initialization
  dataType = ps.getUntrackedParameter<std::string>("dataSample");
  etaSpread = ps.getParameter<double>("etaSpread");
  phiSpread = ps.getParameter<double>("phiSpread");
  chi2Fraction = ps.getParameter<double>("chi2Fraction");
  chi2Spread = ps.getParameter<double>("chi2Spread");
}

MuonTestSummary::~MuonTestSummary(){}

void MuonTestSummary::beginJob(const edm::EventSetup& context){

  metname = "muonTestSummary";
  LogTrace(metname)<<"[MuonTestSummary] Histo booking";

  // book the summary histos
  dbe->setCurrentFolder("Muons/EventInfo"); 
  summaryReport = dbe->bookFloat("reportSummary");
  // Initialize to 0
  summaryReport->Fill(0);

  summaryReportMap = dbe->book2D("reportSummaryMap","Muon Report Summary Map",5,1,6,3,1,4);
  summaryReportMap->setAxisTitle("object monitored",1);
  summaryReportMap->setBinLabel(1,"RecoMuon",1);
  summaryReportMap->setBinLabel(2,"glb mu",1);
  summaryReportMap->setBinLabel(3,"tk mu",1);
  summaryReportMap->setBinLabel(4,"sta mu",1);
  summaryReportMap->setBinLabel(5,"muId",1);
  summaryReportMap->setAxisTitle("test",2);
  summaryReportMap->setBinLabel(1,"meanChi2Red",2);
  summaryReportMap->setBinLabel(2,"eta",2);
  summaryReportMap->setBinLabel(3,"phi",2);
  // Initialize to 0
  for (int x=1; x<=5; x++){
    for (int y=1; y<=3; y++){
      summaryReportMap->Fill(x,y,0);
    }
  }

  dbe->setCurrentFolder("Muons/EventInfo/reportSummaryContents");
  theSummaryContents.push_back(dbe->bookFloat("meanChi2OverDoF"));
  theSummaryContents.push_back(dbe->bookFloat("phiDistribution"));
  theSummaryContents.push_back(dbe->bookFloat("etaDistribution"));
  // Initialize to 0
  for (int i=0; i<int(theSummaryContents.size()); i++){
    theSummaryContents[i]->Fill(0);
  }

}

void MuonTestSummary::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {


  // fill the summaryReportMap
  string muType="globalCosmicMuons";
  doTests(muType, "glb", 2);

  muType="TKTrack";
  doTests(muType, "ctf", 3);

  muType="cosmicMuons";
  doTests(muType, "sta", 4);

  // fill the summaryContents MEs
  double chi2Counter=0;
  double phiCounter=0;
  double etaCounter=0;
  for(int mu=1; mu<=5; mu++){
    if(summaryReportMap->getBinContent(mu,1)==1)
      chi2Counter+=0.2;
    if(summaryReportMap->getBinContent(mu,2)==1)
      phiCounter+=0.2;
    if(summaryReportMap->getBinContent(mu,3)==1)
      etaCounter+=0.2;
  }
  theSummaryContents[0]->Fill(chi2Counter);
  theSummaryContents[1]->Fill(phiCounter);
  theSummaryContents[2]->Fill(etaCounter);

  // fill the summaryReport
  if(chi2Counter==1 && phiCounter==1 && etaCounter==1)
    summaryReport->Fill(1);
  else
    summaryReport->Fill(0);
   
}


void MuonTestSummary::doTests(string type, string AlgoName, int bin){
  

  string path = "Muons/" + type + "/Chi2overDoF_" + AlgoName;
  MonitorElement * chi2Histo = dbe->get(path);

  if(chi2Histo){

    TH1F * chi2Histo_root = chi2Histo->getTH1F();
    int maxBin = chi2Histo_root->GetMaximumBin();
    double fraction = double(chi2Histo_root->Integral(1,maxBin))/double(chi2Histo_root->Integral(maxBin+1,chi2Histo_root->GetNbinsX()));
    if(fraction>(chi2Fraction-chi2Spread) && fraction<(chi2Fraction+chi2Spread))
      summaryReportMap->setBinContent(bin,1,1);
    else
      summaryReportMap->setBinContent(bin,1,0);
  }



  path = "Muons/" + type + "/TrackEta_" + AlgoName;
  MonitorElement * etaHisto = dbe->get(path);
  
  if(etaHisto){
  
    TH1F * etaHisto_root = etaHisto->getTH1F();
    double binSize = (etaHisto_root->GetXaxis()->GetXmax()-etaHisto_root->GetXaxis()->GetXmin())/etaHisto_root->GetNbinsX();
    int binZero = int((0-etaHisto_root->GetXaxis()->GetXmin())/binSize);
    double symmetryFactor = 
      double(etaHisto_root->Integral(1,binZero-1)) / double(etaHisto_root->Integral(binZero,etaHisto_root->GetNbinsX()));
    if(dataType == "ppLike"){
      if (symmetryFactor>(1-etaSpread) && symmetryFactor<(1+etaSpread))
	summaryReportMap->setBinContent(bin,2,1);
      else
	summaryReportMap->setBinContent(bin,2,0);
    }
    if(dataType == "cosmics"){
      if (symmetryFactor>(0.9-etaSpread) && symmetryFactor<(0.9+etaSpread))
	summaryReportMap->setBinContent(bin,2,1);
      else
	summaryReportMap->setBinContent(bin,2,0);
    }
  }


  path = "Muons/" + type + "/TrackPhi_" + AlgoName;
  MonitorElement * phiHisto = dbe->get(path);

  if(phiHisto){
 
    TH1F * phiHisto_root = phiHisto->getTH1F();
    double binSize = (phiHisto_root->GetXaxis()->GetXmax()-phiHisto_root->GetXaxis()->GetXmin())/phiHisto_root->GetNbinsX();
    int binZero = int((0-phiHisto_root->GetXaxis()->GetXmin())/binSize);
    double symmetryFactor = 
      double(phiHisto_root->Integral(binZero+1,phiHisto_root->GetNbinsX())) / double(phiHisto_root->Integral(1,binZero));
  if(dataType == "ppLike"){
      if (symmetryFactor>(1-phiSpread) && symmetryFactor<(1+phiSpread))
	summaryReportMap->setBinContent(bin,3,1);
      else
	summaryReportMap->setBinContent(bin,3,0);
    }
    if(dataType == "cosmics"){
      if (symmetryFactor>(0.01-phiSpread) && symmetryFactor<(0.01+phiSpread))
	summaryReportMap->setBinContent(bin,3,1);
      else
	summaryReportMap->setBinContent(bin,3,0);
    }
  }

}




  

  

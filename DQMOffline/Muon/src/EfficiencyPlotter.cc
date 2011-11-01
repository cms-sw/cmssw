#include "DQMOffline/Muon/src/EfficiencyPlotter.h"

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


EfficiencyPlotter::EfficiencyPlotter(const edm::ParameterSet& ps){
  parameters = ps;
  theDbe = edm::Service<DQMStore>().operator->();
}
EfficiencyPlotter::~EfficiencyPlotter(){}

void EfficiencyPlotter::beginJob(void){
  metname = "muonRecoTest";
  theDbe->setCurrentFolder("Muons/EfficiencyAnalyzer");

  LogTrace(metname)<<"[EfficiencyPlotter] beginJob: Parameters initialization";
 
  // efficiency plot
  etaBin = parameters.getParameter<int>("etaBin");
  etaMin = parameters.getParameter<double>("etaMin");
  etaMax = parameters.getParameter<double>("etaMax");
  etaEfficiency = theDbe->book1D("etaEfficiency_recoMuon", "efficiency vs #eta", etaBin, etaMin, etaMax);

  phiBin = parameters.getParameter<int>("phiBin");
  phiMin = parameters.getParameter<double>("phiMin");
  phiMax = parameters.getParameter<double>("phiMax");
  phiEfficiency = theDbe->book1D("phiEfficiency_recoMuon", "efficiency vs #phi", phiBin, phiMin, phiMax);

  ptBin = parameters.getParameter<int>("ptBin");
  ptMin = parameters.getParameter<double>("ptMin");
  ptMax = parameters.getParameter<double>("ptMax");
  ptEfficiency = theDbe->book1D("ptEfficiency_recoMuon", "#pt_{STA} efficiency", ptBin, ptMin, ptMax);
}
void EfficiencyPlotter::beginRun(Run const& run, EventSetup const& eSetup) {
  LogTrace(metname)<<"[EfficiencyPlotter]: beginRun";
}
void EfficiencyPlotter::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {
  LogTrace(metname)<<"[EfficiencyPlotter]: beginLuminosityBlock";
  // Get the run number
  run = lumiSeg.run();
}
void EfficiencyPlotter::analyze(const edm::Event& e, const edm::EventSetup& context){
  nevents++;
  LogTrace(metname)<< "[EfficiencyPlotter]: "<<nevents<<" events";
}
void EfficiencyPlotter::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {
  //  LogTrace(metname)<<"[EfficiencyPlotter]: endLuminosityBlock, performing the DQM LS client operation";
  // counts number of lumiSegs 
  nLumiSegs = lumiSeg.id().luminosityBlock();
}
void EfficiencyPlotter::endRun(Run const& run, EventSetup const& eSetup) {
  LogTrace(metname)<<"[EfficiencyPlotter]: endRun, performing the DQM end of run client operation";
  
  double efficiency = 0;

  string numpath = "Muons/EfficiencyAnalyzer/Numerator_pt";
  string denpath = "Muons/EfficiencyAnalyzer/Denominator_pt";
  
  MonitorElement *Numerator   = theDbe->get(numpath);
  MonitorElement *Denominator = theDbe->get(numpath);
  if (Numerator && Denominator){
    TH1F *h_numerator   = Numerator->getTH1F();
    TH1F *h_denominator = Denominator->getTH1F();
    
    for(int i=1; i<=ptBin; i++){
      if (h_denominator->GetBinContent(i) == 0)  efficiency = 0;
      else                               	 efficiency = double(h_numerator->GetBinContent(i))/double(h_denominator->GetBinContent(i));
      ptEfficiency->setBinContent(i,efficiency);
    }
  }
  
  numpath = "Muons/EfficiencyAnalyzer/Numerator_eta";
  denpath = "Muons/EfficiencyAnalyzer/Denominator_eta";
  Numerator   = theDbe->get(numpath);
  Denominator = theDbe->get(numpath);
  if (Numerator && Denominator){
    TH1F *h_numerator   = Numerator->getTH1F();
    TH1F *h_denominator = Denominator->getTH1F();
   
    for(int i=1; i<=etaBin; i++){
      if (h_denominator->GetBinContent(i) == 0)  efficiency = 0;
      else                               	 efficiency = double(h_numerator->GetBinContent(i))/double(h_denominator->GetBinContent(i));
      etaEfficiency->setBinContent(i,efficiency);
    }
  }
  
  numpath = "Muons/EfficiencyAnalyzer/Numerator_phi";
  denpath = "Muons/EfficiencyAnalyzer/Denominator_phi";
  Numerator   = theDbe->get(numpath);
  Denominator = theDbe->get(numpath);
  
  if (Numerator && Denominator){
    TH1F *h_numerator   = Numerator->getTH1F();
    TH1F *h_denominator = Denominator->getTH1F();
    
    //    h_numerator->Divide(h_denominator);
    /*  if(h_numerator->GetXaxis()->GetNbins()!=ptBin ||
	h_numerator->GetXaxis()->GetXmax()!=ptBin  ||
	h_numerator->GetXaxis()->GetXmin()!=ptBin){
	LogTrace(metname)<<"[EfficiencyPlotter] wrong histo binning on eta histograms";
	return;
	}
    */
    for(int i=1; i<=phiBin; i++){
      if (h_denominator->GetBinContent(i) == 0)  efficiency = 0;
      else                               	 efficiency = double(h_numerator->GetBinContent(i))/double(h_denominator->GetBinContent(i));
      phiEfficiency->setBinContent(i,efficiency);
    }
  }
}

void EfficiencyPlotter::endJob(){
  LogTrace(metname)<< "[EfficiencyPlotter] endJob called!";
  theDbe->rmdir("Muons/EfficiencyAnalyzer");
}
  

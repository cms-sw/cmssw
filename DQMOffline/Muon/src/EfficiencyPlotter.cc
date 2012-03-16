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
#include "TH1F.h"


using namespace edm;
using namespace std;


EfficiencyPlotter::EfficiencyPlotter(const edm::ParameterSet& ps){
  parameters = ps;
  theDbe = edm::Service<DQMStore>().operator->();
}
EfficiencyPlotter::~EfficiencyPlotter(){}

void EfficiencyPlotter::beginJob(void){
  //metname = "muonRecoTest";
  
  metname = "EfficiencyAnalyzer";
  theDbe->setCurrentFolder("Muons/EfficiencyAnalyzer");

  LogTrace(metname)<<"[EfficiencyPlotter] beginJob: Parameters initialization";
 
  // efficiency plot
  etaBin = parameters.getParameter<int>("etaBin");
  etaMin = parameters.getParameter<double>("etaMin");
  etaMax = parameters.getParameter<double>("etaMax");
  h_eff_eta_TightMu = theDbe->book1D("Eff_eta_TightMu", "TightMu Efficiency vs #eta", etaBin, etaMin, etaMax);
  h_eff_hp_eta_TightMu = theDbe->book1D("Eff_hp_eta_TightMu", "High Pt TightMu Efficiency vs #eta", etaBin, etaMin, etaMax);

  phiBin = parameters.getParameter<int>("phiBin");
  phiMin = parameters.getParameter<double>("phiMin");
  phiMax = parameters.getParameter<double>("phiMax");
  h_eff_phi_TightMu = theDbe->book1D("Eff_phi_TightMu", "TightMu Efficiency vs #phi", phiBin, phiMin, phiMax);

  ptBin = parameters.getParameter<int>("ptBin");
  ptMin = parameters.getParameter<double>("ptMin");
  ptMax = parameters.getParameter<double>("ptMax");

  vtxBin = parameters.getParameter<int>("vtxBin");
  vtxMin = parameters.getParameter<int>("vtxMin");
  vtxMax = parameters.getParameter<int>("vtxMax");

  h_eff_pt_TightMu = theDbe->book1D("Eff_pt_TightMu", "TightMu Efficiency vs Pt", ptBin, ptMin, ptMax);
  h_eff_pt_barrel_TightMu = theDbe->book1D("Eff_pt_barrel_TightMu", "Barrel: TightMu Efficiency vs Pt", ptBin, ptMin, ptMax);
  h_eff_pt_endcap_TightMu = theDbe->book1D("Eff_pt_endcap_TightMu", "Endcap: TightMu Efficiency vs Pt", ptBin, ptMin, ptMax);

  h_eff_pt_detIsoTightMu = theDbe->book1D("Eff_pt_detIsoTightMu", "detIsoTightMu Efficiency vs Pt", ptBin, ptMin, ptMax);
  h_eff_pt_barrel_detIsoTightMu = theDbe->book1D("Eff_pt_barrel_detIsoTightMu", "Barrel: detIsoTightMu Efficiency vs Pt", ptBin, ptMin, ptMax);
  h_eff_pt_endcap_detIsoTightMu = theDbe->book1D("Eff_pt_endcap_detIsoTightMu", "Endcap: detIsoTightMu Efficiency vs Pt", ptBin, ptMin, ptMax);

  h_eff_pt_pfIsoTightMu = theDbe->book1D("Eff_pt_pfIsoTightMu", "pfIsoTightMu Efficiency vs Pt", ptBin, ptMin, ptMax);
  h_eff_pt_barrel_pfIsoTightMu = theDbe->book1D("Eff_pt_barrel_pfIsoTightMu", "Barrel: pfIsoTightMu Efficiency vs Pt", ptBin, ptMin, ptMax);
  h_eff_pt_endcap_pfIsoTightMu = theDbe->book1D("Eff_pt_endcap_pfIsoTightMu", "Endcap: pfIsoTightMu Efficiency vs Pt", ptBin, ptMin, ptMax);



  h_eff_vtx_detIsoTightMu = theDbe->book1D("Eff_vtx_detIsoTightMu", "detIsoTightMu Efficiency vs nVtx", vtxBin, vtxMin, vtxMax);
  h_eff_vtx_pfIsoTightMu = theDbe->book1D("Eff_vtx_pfIsoTightMu", "pfIsoTightMu Efficiency vs nVtx", vtxBin, vtxMin, vtxMax);




  h_eff_eta_TightMu->setEfficiencyFlag();
  h_eff_hp_eta_TightMu->setEfficiencyFlag();
  h_eff_phi_TightMu->setEfficiencyFlag();
  h_eff_pt_TightMu->setEfficiencyFlag();
  h_eff_pt_barrel_TightMu->setEfficiencyFlag();
  h_eff_pt_endcap_TightMu->setEfficiencyFlag();
  h_eff_pt_detIsoTightMu->setEfficiencyFlag();
  h_eff_pt_barrel_detIsoTightMu->setEfficiencyFlag();
  h_eff_pt_endcap_detIsoTightMu->setEfficiencyFlag();
  h_eff_pt_pfIsoTightMu->setEfficiencyFlag();
  h_eff_pt_barrel_pfIsoTightMu->setEfficiencyFlag();
  h_eff_pt_endcap_pfIsoTightMu->setEfficiencyFlag();


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
  
 

  /// --- Tight Muon efficiency vs muon Pt 

  string numpath_pt = "Muons/EfficiencyAnalyzer/passProbes_TightMu_pt";
  string denpath_pt = "Muons/EfficiencyAnalyzer/allProbes_pt";
  
  MonitorElement *Numerator_pt   = theDbe->get(numpath_pt);
  MonitorElement *Denominator_pt = theDbe->get(denpath_pt);
  
  if (Numerator_pt && Denominator_pt){
 
    TH1F *h_numerator_pt   = Numerator_pt->getTH1F();
    TH1F *h_denominator_pt = Denominator_pt->getTH1F();

    TH1F *h_eff_pt = h_eff_pt_TightMu->getTH1F();
    
    h_eff_pt->Sumw2(); 
    
    h_eff_pt->Divide(h_numerator_pt, h_denominator_pt, 1., 1., "B");

  }

  /// --- Tight Muon efficiency vs muon Pt [Barrel]

  string numpath_barrel_pt = "Muons/EfficiencyAnalyzer/passProbes_TightMu_barrel_pt";
  string denpath_barrel_pt = "Muons/EfficiencyAnalyzer/allProbes_barrel_pt";
  
  MonitorElement *Numerator_barrel_pt   = theDbe->get(numpath_barrel_pt);
  MonitorElement *Denominator_barrel_pt = theDbe->get(denpath_barrel_pt);
  
  if (Numerator_barrel_pt && Denominator_barrel_pt){
 
    TH1F *h_numerator_barrel_pt   = Numerator_barrel_pt->getTH1F();
    TH1F *h_denominator_barrel_pt = Denominator_barrel_pt->getTH1F();

    TH1F *h_eff_barrel_pt = h_eff_pt_barrel_TightMu->getTH1F();
    
    h_eff_barrel_pt->Sumw2(); 
    
    h_eff_barrel_pt->Divide(h_numerator_barrel_pt, h_denominator_barrel_pt, 1., 1., "B");

  }

  /// --- Tight Muon efficiency vs muon Pt [Endcap]

  string numpath_endcap_pt = "Muons/EfficiencyAnalyzer/passProbes_TightMu_endcap_pt";
  string denpath_endcap_pt = "Muons/EfficiencyAnalyzer/allProbes_endcap_pt";
  
  MonitorElement *Numerator_endcap_pt   = theDbe->get(numpath_endcap_pt);
  MonitorElement *Denominator_endcap_pt = theDbe->get(denpath_endcap_pt);
  
  if (Numerator_endcap_pt && Denominator_endcap_pt){
 
    TH1F *h_numerator_endcap_pt   = Numerator_endcap_pt->getTH1F();
    TH1F *h_denominator_endcap_pt = Denominator_endcap_pt->getTH1F();

    TH1F *h_eff_endcap_pt = h_eff_pt_endcap_TightMu->getTH1F();
    
    h_eff_endcap_pt->Sumw2(); 
    
    h_eff_endcap_pt->Divide(h_numerator_endcap_pt, h_denominator_endcap_pt, 1., 1., "B");

  }

  /// --- Tight Muon efficiency vs muon Eta

  string numpath_eta = "Muons/EfficiencyAnalyzer/passProbes_TightMu_eta";
  string denpath_eta = "Muons/EfficiencyAnalyzer/allProbes_eta";
  
  MonitorElement *Numerator_eta   = theDbe->get(numpath_eta);
  MonitorElement *Denominator_eta = theDbe->get(denpath_eta);
  
  if (Numerator_eta && Denominator_eta){
 
    TH1F *h_numerator_eta   = Numerator_eta->getTH1F();
    TH1F *h_denominator_eta = Denominator_eta->getTH1F();

    TH1F *h_eff_eta = h_eff_eta_TightMu->getTH1F();
    
    h_eff_eta->Sumw2(); 
    
    h_eff_eta->Divide(h_numerator_eta, h_denominator_eta, 1., 1., "B");

  }

  /// --- Tight Muon efficiency vs muon Eta [ pt > 20 ] 

  string numpath_hp_eta = "Muons/EfficiencyAnalyzer/passProbes_TightMu_hp_eta";
  string denpath_hp_eta = "Muons/EfficiencyAnalyzer/allProbes_hp_eta";
  
  MonitorElement *Numerator_hp_eta   = theDbe->get(numpath_hp_eta);
  MonitorElement *Denominator_hp_eta = theDbe->get(denpath_hp_eta);
  
  if (Numerator_hp_eta && Denominator_hp_eta){
 
    TH1F *h_numerator_hp_eta   = Numerator_hp_eta->getTH1F();
    TH1F *h_denominator_hp_eta = Denominator_hp_eta->getTH1F();

    TH1F *h_eff_hp_eta = h_eff_hp_eta_TightMu->getTH1F();
    
    h_eff_hp_eta->Sumw2(); 
    
    h_eff_hp_eta->Divide(h_numerator_hp_eta, h_denominator_hp_eta, 1., 1., "B");

  }

  /// --- Tight Muon efficiency vs muon Phi 

  string numpath_phi = "Muons/EfficiencyAnalyzer/passProbes_TightMu_phi";
  string denpath_phi = "Muons/EfficiencyAnalyzer/allProbes_phi";
  
  MonitorElement *Numerator_phi   = theDbe->get(numpath_phi);
  MonitorElement *Denominator_phi = theDbe->get(denpath_phi);
  
  if (Numerator_phi && Denominator_phi){
 
    TH1F *h_numerator_phi   = Numerator_phi->getTH1F();
    TH1F *h_denominator_phi = Denominator_phi->getTH1F();

    TH1F *h_eff_phi = h_eff_phi_TightMu->getTH1F();
    
    h_eff_phi->Sumw2(); 
    
    h_eff_phi->Divide(h_numerator_phi, h_denominator_phi, 1., 1., "B");

  }


  /// --- Tight Muon + Detector Isolation  efficiency vs muon Pt

  string numpath_detIso_pt = "Muons/EfficiencyAnalyzer/passProbes_detIsoTightMu_pt";
  string denpath_detIso_pt = "Muons/EfficiencyAnalyzer/allProbes_TightMu_pt";
  
  MonitorElement *Numerator_detIso_pt   = theDbe->get(numpath_detIso_pt);
  MonitorElement *Denominator_detIso_pt = theDbe->get(denpath_detIso_pt);
  
  if (Numerator_detIso_pt && Denominator_detIso_pt){
 
    TH1F *h_numerator_detIso_pt   = Numerator_detIso_pt->getTH1F();
    TH1F *h_denominator_detIso_pt = Denominator_detIso_pt->getTH1F();

    TH1F *h_eff_detIso_pt = h_eff_pt_detIsoTightMu->getTH1F();
    
    h_eff_detIso_pt->Sumw2(); 
    
    h_eff_detIso_pt->Divide(h_numerator_detIso_pt, h_denominator_detIso_pt, 1., 1., "B");

  }


  /// --- Tight Muon + Detector Isolation  efficiency vs muon Pt [Barrel] 

  string numpath_detIso_barrel_pt = "Muons/EfficiencyAnalyzer/passProbes_barrel_detIsoTightMu_pt";
  string denpath_detIso_barrel_pt = "Muons/EfficiencyAnalyzer/allProbes_barrel_TightMu_pt";
  
  MonitorElement *Numerator_detIso_barrel_pt   = theDbe->get(numpath_detIso_barrel_pt);
  MonitorElement *Denominator_detIso_barrel_pt = theDbe->get(denpath_detIso_barrel_pt);
  
  if (Numerator_detIso_barrel_pt && Denominator_detIso_barrel_pt){
 
    TH1F *h_numerator_detIso_barrel_pt   = Numerator_detIso_barrel_pt->getTH1F();
    TH1F *h_denominator_detIso_barrel_pt = Denominator_detIso_barrel_pt->getTH1F();

    TH1F *h_eff_detIso_barrel_pt = h_eff_pt_barrel_detIsoTightMu->getTH1F();
    
    h_eff_detIso_barrel_pt->Sumw2(); 
    
    h_eff_detIso_barrel_pt->Divide(h_numerator_detIso_barrel_pt, h_denominator_detIso_barrel_pt, 1., 1., "B");

  }


  /// --- Tight Muon + Detector Isolation  efficiency vs muon Pt [Endcap] 

  string numpath_detIso_endcap_pt = "Muons/EfficiencyAnalyzer/passProbes_endcap_detIsoTightMu_pt";
  string denpath_detIso_endcap_pt = "Muons/EfficiencyAnalyzer/allProbes_endcap_TightMu_pt";
  
  MonitorElement *Numerator_detIso_endcap_pt   = theDbe->get(numpath_detIso_endcap_pt);
  MonitorElement *Denominator_detIso_endcap_pt = theDbe->get(denpath_detIso_endcap_pt);
  
  if (Numerator_detIso_endcap_pt && Denominator_detIso_endcap_pt){
 
    TH1F *h_numerator_detIso_endcap_pt   = Numerator_detIso_endcap_pt->getTH1F();
    TH1F *h_denominator_detIso_endcap_pt = Denominator_detIso_endcap_pt->getTH1F();

    TH1F *h_eff_detIso_endcap_pt = h_eff_pt_endcap_detIsoTightMu->getTH1F();
    
    h_eff_detIso_endcap_pt->Sumw2(); 
    
    h_eff_detIso_endcap_pt->Divide(h_numerator_detIso_endcap_pt, h_denominator_detIso_endcap_pt, 1., 1., "B");

  }


 /// --- Tight Muon + PF Isolation  efficiency vs muon Pt

  string numpath_pfIso_pt = "Muons/EfficiencyAnalyzer/passProbes_pfIsoTightMu_pt";
  string denpath_pfIso_pt = "Muons/EfficiencyAnalyzer/allProbes_TightMu_pt";
  
  MonitorElement *Numerator_pfIso_pt   = theDbe->get(numpath_pfIso_pt);
  MonitorElement *Denominator_pfIso_pt = theDbe->get(denpath_pfIso_pt);
  
  if (Numerator_pfIso_pt && Denominator_pfIso_pt){
 
    TH1F *h_numerator_pfIso_pt   = Numerator_pfIso_pt->getTH1F();
    TH1F *h_denominator_pfIso_pt = Denominator_pfIso_pt->getTH1F();

    TH1F *h_eff_pfIso_pt = h_eff_pt_pfIsoTightMu->getTH1F();
    
    h_eff_pfIso_pt->Sumw2(); 
    
    h_eff_pfIso_pt->Divide(h_numerator_pfIso_pt, h_denominator_pfIso_pt, 1., 1., "B");

  }


  /// --- Tight Muon + PF Isolation  efficiency vs muon Pt [Barrel] 

  string numpath_pfIso_barrel_pt = "Muons/EfficiencyAnalyzer/passProbes_barrel_pfIsoTightMu_pt";
  string denpath_pfIso_barrel_pt = "Muons/EfficiencyAnalyzer/allProbes_barrel_TightMu_pt";
  
  MonitorElement *Numerator_pfIso_barrel_pt   = theDbe->get(numpath_pfIso_barrel_pt);
  MonitorElement *Denominator_pfIso_barrel_pt = theDbe->get(denpath_pfIso_barrel_pt);
  
  if (Numerator_pfIso_barrel_pt && Denominator_pfIso_barrel_pt){
 
    TH1F *h_numerator_pfIso_barrel_pt   = Numerator_pfIso_barrel_pt->getTH1F();
    TH1F *h_denominator_pfIso_barrel_pt = Denominator_pfIso_barrel_pt->getTH1F();

    TH1F *h_eff_pfIso_barrel_pt = h_eff_pt_barrel_pfIsoTightMu->getTH1F();
    
    h_eff_pfIso_barrel_pt->Sumw2(); 
    
    h_eff_pfIso_barrel_pt->Divide(h_numerator_pfIso_barrel_pt, h_denominator_pfIso_barrel_pt, 1., 1., "B");

  }


  /// --- Tight Muon + PF Isolation  efficiency vs muon Pt [Endcap] 

  string numpath_pfIso_endcap_pt = "Muons/EfficiencyAnalyzer/passProbes_endcap_pfIsoTightMu_pt";
  string denpath_pfIso_endcap_pt = "Muons/EfficiencyAnalyzer/allProbes_endcap_TightMu_pt";
  
  MonitorElement *Numerator_pfIso_endcap_pt   = theDbe->get(numpath_pfIso_endcap_pt);
  MonitorElement *Denominator_pfIso_endcap_pt = theDbe->get(denpath_pfIso_endcap_pt);
  
  if (Numerator_pfIso_endcap_pt && Denominator_pfIso_endcap_pt){
 
    TH1F *h_numerator_pfIso_endcap_pt   = Numerator_pfIso_endcap_pt->getTH1F();
    TH1F *h_denominator_pfIso_endcap_pt = Denominator_pfIso_endcap_pt->getTH1F();

    TH1F *h_eff_pfIso_endcap_pt = h_eff_pt_endcap_pfIsoTightMu->getTH1F();
    
    h_eff_pfIso_endcap_pt->Sumw2(); 
    
    h_eff_pfIso_endcap_pt->Divide(h_numerator_pfIso_endcap_pt, h_denominator_pfIso_endcap_pt, 1., 1., "B");

  }


  /// --- Tight Muon + PF Isolation  efficiency vs muon number of reco Vertex

  string numpath_pfIso_nvtx = "Muons/EfficiencyAnalyzer/passProbes_pfIsoTightMu_nVtx";
  string denpath_pfIso_nvtx = "Muons/EfficiencyAnalyzer/allProbes_TightMu_nVtx";
  
  MonitorElement *Numerator_pfIso_nvtx   = theDbe->get(numpath_pfIso_nvtx);
  MonitorElement *Denominator_pfIso_nvtx = theDbe->get(denpath_pfIso_nvtx);
  
  if (Numerator_pfIso_nvtx && Denominator_pfIso_nvtx){
 
    TH1F *h_numerator_pfIso_nvtx   = Numerator_pfIso_nvtx->getTH1F();
    TH1F *h_denominator_pfIso_nvtx = Denominator_pfIso_nvtx->getTH1F();

    TH1F *h_eff_pfIso_nvtx = h_eff_vtx_pfIsoTightMu->getTH1F();
    
    h_eff_pfIso_nvtx->Sumw2(); 
    
    h_eff_pfIso_nvtx->Divide(h_numerator_pfIso_nvtx, h_denominator_pfIso_nvtx, 1., 1., "B");

  }



 /// --- Tight Muon + detector-Based Isolation  efficiency vs muon number of reco Vertex

  string numpath_detIso_nvtx = "Muons/EfficiencyAnalyzer/passProbes_detIsoTightMu_nVtx";
  string denpath_detIso_nvtx = "Muons/EfficiencyAnalyzer/allProbes_TightMu_nVtx";
  
  MonitorElement *Numerator_detIso_nvtx   = theDbe->get(numpath_detIso_nvtx);
  MonitorElement *Denominator_detIso_nvtx = theDbe->get(denpath_detIso_nvtx);
  
  if (Numerator_detIso_nvtx && Denominator_detIso_nvtx){
 
    TH1F *h_numerator_detIso_nvtx   = Numerator_detIso_nvtx->getTH1F();
    TH1F *h_denominator_detIso_nvtx = Denominator_detIso_nvtx->getTH1F();

    TH1F *h_eff_detIso_nvtx = h_eff_vtx_detIsoTightMu->getTH1F();
    
    h_eff_detIso_nvtx->Sumw2(); 
    
    h_eff_detIso_nvtx->Divide(h_numerator_detIso_nvtx, h_denominator_detIso_nvtx, 1., 1., "B");

  }





}


void EfficiencyPlotter::endJob(){
  LogTrace(metname)<< "[EfficiencyPlotter] endJob called!";
  theDbe->rmdir("Muons/EfficiencyAnalyzer");
}
  

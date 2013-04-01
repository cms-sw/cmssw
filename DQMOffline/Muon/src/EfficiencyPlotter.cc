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

//#define DEBUG

EfficiencyPlotter::EfficiencyPlotter(const edm::ParameterSet& ps){
#ifdef DEBUG
  cout << "EfficiencyPlotter(): Constructor " << endl;
#endif
  parameters = ps;
  theDbe = edm::Service<DQMStore>().operator->();
}
EfficiencyPlotter::~EfficiencyPlotter(){}

void EfficiencyPlotter::beginJob(void){
#ifdef DEBUG
  cout << "EfficiencyPlotter::beginJob " << endl;
#endif
  
  theDbe->setCurrentFolder("Muons/EfficiencyAnalyzer");
  
  metname = "EfficiencyAnalyzer";
  LogTrace(metname)<<"[EfficiencyPlotter] beginJob: Parameters initialization";
 
  // efficiency plot
  etaBin = parameters.getParameter<int>("etaBin");
  etaMin = parameters.getParameter<double>("etaMin");
  etaMax = parameters.getParameter<double>("etaMax");
  
  phiBin = parameters.getParameter<int>("phiBin");
  phiMin = parameters.getParameter<double>("phiMin");
  phiMax = parameters.getParameter<double>("phiMax");

  ptBin = parameters.getParameter<int>("ptBin");
  ptMin = parameters.getParameter<double>("ptMin");
  ptMax = parameters.getParameter<double>("ptMax");

  vtxBin = parameters.getParameter<int>("vtxBin");
  vtxMin = parameters.getParameter<double>("vtxMin");
  vtxMax = parameters.getParameter<double>("vtxMax");

  h_eff_eta_TightMu          = theDbe->book1D("Eff_eta_TightMu",          "TightMu Eff. vs #eta",               etaBin, etaMin, etaMax);
  h_eff_hp_eta_TightMu       = theDbe->book1D("Eff_hp_eta_TightMu",       "High Pt TightMu Eff. vs #eta",       etaBin, etaMin, etaMax);
  h_eff_phi_TightMu          = theDbe->book1D("Eff_phi_TightMu",          "TightMu Eff. vs #phi",               phiBin, phiMin, phiMax);
  h_eff_pt_TightMu           = theDbe->book1D("Eff_pt_TightMu",           "TightMu Eff. vs Pt",                 ptBin, ptMin, ptMax);
  h_eff_pt_EB_TightMu        = theDbe->book1D("Eff_pt_EB_TightMu",        "Barrel: TightMu Eff. vs Pt",         ptBin, ptMin, ptMax);
  h_eff_pt_EE_TightMu        = theDbe->book1D("Eff_pt_EE_TightMu",        "Endcap: TightMu Eff. vs Pt",         ptBin, ptMin, ptMax);
  h_eff_pt_detIsoTightMu     = theDbe->book1D("Eff_pt_detIsoTightMu",     "detIsoTightMu Efficiency vs Pt",     ptBin, ptMin, ptMax);
  h_eff_pt_EB_detIsoTightMu  = theDbe->book1D("Eff_pt_EB_detIsoTightMu",  "Barrel: detIsoTightMu Eff. vs Pt",   ptBin, ptMin, ptMax);
  h_eff_pt_EE_detIsoTightMu  = theDbe->book1D("Eff_pt_EE_detIsoTightMu",  "Endcap: detIsoTightMu Eff. vs Pt",   ptBin, ptMin, ptMax);
  h_eff_pt_pfIsoTightMu      = theDbe->book1D("Eff_pt_pfIsoTightMu",      "pfIsoTightMu Eff. vs Pt",            ptBin, ptMin, ptMax);
  h_eff_pt_EB_pfIsoTightMu   = theDbe->book1D("Eff_pt_EB_pfIsoTightMu",   "Barrel: pfIsoTightMu Eff. vs Pt",    ptBin, ptMin, ptMax);
  h_eff_pt_EE_pfIsoTightMu   = theDbe->book1D("Eff_pt_EE_pfIsoTightMu",   "Endcap: pfIsoTightMu Eff. vs Pt",    ptBin, ptMin, ptMax);
  h_eff_vtx_detIsoTightMu    = theDbe->book1D("Eff_vtx_detIsoTightMu",    "detIsoTightMu Eff. vs nVtx",         vtxBin, vtxMin, vtxMax);
  h_eff_vtx_pfIsoTightMu     = theDbe->book1D("Eff_vtx_pfIsoTightMu",     "pfIsoTightMu Eff. vs nVtx",          vtxBin, vtxMin, vtxMax);
  h_eff_vtx_EB_detIsoTightMu = theDbe->book1D("Eff_vtx_EB_detIsoTightMu", "Barrel: detIsoTightMu Eff. vs nVtx", vtxBin, vtxMin, vtxMax);
  h_eff_vtx_EB_pfIsoTightMu  = theDbe->book1D("Eff_vtx_EB_pfIsoTightMu",  "Barrel: pfIsoTightMu Eff. vs nVtx",  vtxBin, vtxMin, vtxMax);
  h_eff_vtx_EE_detIsoTightMu = theDbe->book1D("Eff_vtx_EE_detIsoTightMu", "Endcap: detIsoTightMu Eff. vs nVtx", vtxBin, vtxMin, vtxMax);
  h_eff_vtx_EE_pfIsoTightMu  = theDbe->book1D("Eff_vtx_EE_pfIsoTightMu",  "Endcap: pfIsoTightMu Eff. vs nVtx",  vtxBin, vtxMin, vtxMax);

  h_eff_pt_pfIsodBTightMu      = theDbe->book1D("Eff_pt_pfIsodBTightMu",      "pfIsoTightMu (deltaBeta) Eff. vs Pt",            ptBin, ptMin, ptMax);
  h_eff_pt_EB_pfIsodBTightMu   = theDbe->book1D("Eff_pt_EB_pfIsodBTightMu",   "Barrel: pfIsoTightMu (deltaBeta) Eff. vs Pt",    ptBin, ptMin, ptMax);
  h_eff_pt_EE_pfIsodBTightMu   = theDbe->book1D("Eff_pt_EE_pfIsodBTightMu",   "Endcap: pfIsoTightMu (deltaBeta) Eff. vs Pt",    ptBin, ptMin, ptMax);
  h_eff_vtx_pfIsodBTightMu     = theDbe->book1D("Eff_vtx_pfIsodBTightMu",     "pfIsoTightMu (deltaBeta) Eff. vs nVtx",          vtxBin, vtxMin, vtxMax);
  h_eff_vtx_EB_pfIsodBTightMu  = theDbe->book1D("Eff_vtx_EB_pfIsodBTightMu",  "Barrel: pfIsoTightMu (deltaBeta) Eff. vs nVtx",  vtxBin, vtxMin, vtxMax);
  h_eff_vtx_EE_pfIsodBTightMu  = theDbe->book1D("Eff_vtx_EE_pfIsodBTightMu",  "Endcap: pfIsoTightMu (deltaBeta) Eff. vs nVtx",  vtxBin, vtxMin, vtxMax);



  // This prevents this ME to be normalized when drawn into the GUI
  h_eff_eta_TightMu         ->setEfficiencyFlag();
  h_eff_hp_eta_TightMu      ->setEfficiencyFlag();
  h_eff_phi_TightMu         ->setEfficiencyFlag();
  h_eff_pt_TightMu          ->setEfficiencyFlag();
  h_eff_pt_EB_TightMu       ->setEfficiencyFlag();
  h_eff_pt_EE_TightMu       ->setEfficiencyFlag();
  h_eff_pt_detIsoTightMu    ->setEfficiencyFlag();
  h_eff_pt_EB_detIsoTightMu ->setEfficiencyFlag();
  h_eff_pt_EE_detIsoTightMu ->setEfficiencyFlag();
  h_eff_pt_pfIsoTightMu     ->setEfficiencyFlag();
  h_eff_pt_EB_pfIsoTightMu  ->setEfficiencyFlag();
  h_eff_pt_EE_pfIsoTightMu  ->setEfficiencyFlag();
  h_eff_vtx_detIsoTightMu   ->setEfficiencyFlag();
  h_eff_vtx_pfIsoTightMu    ->setEfficiencyFlag();
  h_eff_vtx_EB_detIsoTightMu->setEfficiencyFlag();
  h_eff_vtx_EB_pfIsoTightMu ->setEfficiencyFlag();
  h_eff_vtx_EE_detIsoTightMu->setEfficiencyFlag();
  h_eff_vtx_EE_pfIsoTightMu ->setEfficiencyFlag();

  h_eff_pt_pfIsodBTightMu   ->setEfficiencyFlag();
  h_eff_pt_EB_pfIsodBTightMu ->setEfficiencyFlag();
  h_eff_pt_EE_pfIsodBTightMu ->setEfficiencyFlag();
  h_eff_vtx_pfIsodBTightMu   ->setEfficiencyFlag();
  h_eff_vtx_EB_pfIsodBTightMu ->setEfficiencyFlag();
  h_eff_vtx_EE_pfIsodBTightMu ->setEfficiencyFlag();



  // AXIS TITLES....
  h_eff_hp_eta_TightMu      ->setAxisTitle("#eta",         1);  
  h_eff_eta_TightMu         ->setAxisTitle("#eta",         1);  
  h_eff_phi_TightMu         ->setAxisTitle("#phi",         1);  
  h_eff_pt_TightMu          ->setAxisTitle("p_{T} (GeV)",  1);  
  h_eff_pt_EB_TightMu       ->setAxisTitle("p_{T} (GeV)",  1);  
  h_eff_pt_EE_TightMu       ->setAxisTitle("p_{T} (GeV)",  1);  
  h_eff_pt_detIsoTightMu    ->setAxisTitle("p_{T} (GeV)",  1);  
  h_eff_pt_EB_detIsoTightMu ->setAxisTitle("p_{T} (GeV)",  1);  
  h_eff_pt_EE_detIsoTightMu ->setAxisTitle("p_{T} (GeV)",  1);  
  h_eff_pt_pfIsoTightMu     ->setAxisTitle("p_{T} (GeV)",  1);  
  h_eff_pt_EB_pfIsoTightMu  ->setAxisTitle("p_{T} (GeV)",  1);  
  h_eff_pt_EE_pfIsoTightMu  ->setAxisTitle("p_{T} (GeV)",  1);  
  h_eff_vtx_detIsoTightMu   ->setAxisTitle("Number of PV", 1);
  h_eff_vtx_pfIsoTightMu    ->setAxisTitle("Number of PV", 1);
  h_eff_vtx_EB_detIsoTightMu->setAxisTitle("Number of PV", 1);
  h_eff_vtx_EB_pfIsoTightMu ->setAxisTitle("Number of PV", 1);
  h_eff_vtx_EE_detIsoTightMu->setAxisTitle("Number of PV", 1);
  h_eff_vtx_EE_pfIsoTightMu ->setAxisTitle("Number of PV", 1);

  h_eff_pt_pfIsodBTightMu     ->setAxisTitle("p_{T} (GeV)",  1); 
  h_eff_pt_EB_pfIsodBTightMu  ->setAxisTitle("p_{T} (GeV)",  1);  
  h_eff_pt_EE_pfIsodBTightMu  ->setAxisTitle("p_{T} (GeV)",  1);   
  h_eff_vtx_pfIsodBTightMu    ->setAxisTitle("Number of PV", 1);
  h_eff_vtx_EB_pfIsodBTightMu ->setAxisTitle("Number of PV", 1);
  h_eff_vtx_EE_pfIsodBTightMu ->setAxisTitle("Number of PV", 1);

  
//   h_eff_eta_TightMu         ->setAxisTitle("Tight Mu Eff.",      2);  
//   h_eff_hp_eta_TightMu      ->setAxisTitle("High p_{T} Mu Eff.", 2);  
//   h_eff_phi_TightMu         ->setAxisTitle("Tight Mu Eff.",      2);  
//   h_eff_pt_TightMu          ->setAxisTitle("Tight Mu Eff.",      2);  
//   h_eff_pt_EB_TightMu       ->setAxisTitle("Tight Mu Eff.",      2);  
//   h_eff_pt_EE_TightMu       ->setAxisTitle("Tight Mu Eff.",      2);  
//   h_eff_pt_detIsoTightMu    ->setAxisTitle("Tight Mu Eff.",      2);  
//   h_eff_pt_EB_detIsoTightMu ->setAxisTitle("Tight Mu Eff.",      2);  
//   h_eff_pt_EE_detIsoTightMu ->setAxisTitle("Tight Mu Eff.",      2);  
//   h_eff_pt_pfIsoTightMu     ->setAxisTitle("Tight Mu Eff.",      2);  
//   h_eff_pt_EB_pfIsoTightMu  ->setAxisTitle("Tight Mu Eff.",      2);  
//   h_eff_pt_EE_pfIsoTightMu  ->setAxisTitle("Tight Mu Eff.",      2);  
//   h_eff_vtx_detIsoTightMu   ->setAxisTitle("Tight Mu Eff.",      2);
//   h_eff_vtx_pfIsoTightMu    ->setAxisTitle("Tight Mu Eff.",      2);
//   h_eff_vtx_EB_detIsoTightMu->setAxisTitle("Tight Mu Eff.",      2);
//   h_eff_vtx_EB_pfIsoTightMu ->setAxisTitle("Tight Mu Eff.",      2);
//   h_eff_vtx_EE_detIsoTightMu->setAxisTitle("Tight Mu Eff.",      2);
//   h_eff_vtx_EE_pfIsoTightMu ->setAxisTitle("Tight Mu Eff.",      2);
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
    TH1F *h_eff_pt         = h_eff_pt_TightMu->getTH1F();
    
    if (h_eff_pt->GetSumw2N() == 0) h_eff_pt->Sumw2();  
    h_eff_pt->Divide(h_numerator_pt, h_denominator_pt, 1., 1., "B");
  }
  
  /// --- Tight Muon efficiency vs muon Pt [EB]
  string numpath_EB_pt = "Muons/EfficiencyAnalyzer/passProbes_TightMu_EB_pt";
  string denpath_EB_pt = "Muons/EfficiencyAnalyzer/allProbes_EB_pt";
  
  MonitorElement *Numerator_EB_pt   = theDbe->get(numpath_EB_pt);
  MonitorElement *Denominator_EB_pt = theDbe->get(denpath_EB_pt);
  
  if (Numerator_EB_pt && Denominator_EB_pt){
    TH1F *h_numerator_EB_pt   = Numerator_EB_pt->getTH1F();
    TH1F *h_denominator_EB_pt = Denominator_EB_pt->getTH1F();
    TH1F *h_eff_EB_pt         = h_eff_pt_EB_TightMu->getTH1F();
    
    if (h_eff_EB_pt->GetSumw2N() == 0) h_eff_EB_pt->Sumw2();  
    h_eff_EB_pt->Divide(h_numerator_EB_pt, h_denominator_EB_pt, 1., 1., "B");
  }

  /// --- Tight Muon efficiency vs muon Pt [EE]
  string numpath_EE_pt = "Muons/EfficiencyAnalyzer/passProbes_TightMu_EE_pt";
  string denpath_EE_pt = "Muons/EfficiencyAnalyzer/allProbes_EE_pt";
  
  MonitorElement *Numerator_EE_pt   = theDbe->get(numpath_EE_pt);
  MonitorElement *Denominator_EE_pt = theDbe->get(denpath_EE_pt);
  
  if (Numerator_EE_pt && Denominator_EE_pt){
    TH1F *h_numerator_EE_pt   = Numerator_EE_pt->getTH1F();
    TH1F *h_denominator_EE_pt = Denominator_EE_pt->getTH1F();
    TH1F *h_eff_EE_pt         = h_eff_pt_EE_TightMu->getTH1F();
    
    if (h_eff_EE_pt->GetSumw2N() == 0) h_eff_EE_pt->Sumw2();  
    h_eff_EE_pt->Divide(h_numerator_EE_pt, h_denominator_EE_pt, 1., 1., "B");
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
    
    if (h_eff_eta->GetSumw2N() == 0) h_eff_eta->Sumw2();  
    
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
    
    if (h_eff_hp_eta->GetSumw2N() == 0) h_eff_hp_eta->Sumw2();  
    
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
    
    if (h_eff_phi->GetSumw2N() == 0) h_eff_phi->Sumw2();  
    
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
    
    if (h_eff_detIso_pt->GetSumw2N() == 0) h_eff_detIso_pt->Sumw2();  
    h_eff_detIso_pt->Divide(h_numerator_detIso_pt, h_denominator_detIso_pt, 1., 1., "B");
    
  }


  /// --- Tight Muon + Detector Isolation  efficiency vs muon Pt [EB] 
  string numpath_detIso_EB_pt = "Muons/EfficiencyAnalyzer/passProbes_EB_detIsoTightMu_pt";
  string denpath_detIso_EB_pt = "Muons/EfficiencyAnalyzer/allProbes_EB_TightMu_pt";
  
  MonitorElement *Numerator_detIso_EB_pt   = theDbe->get(numpath_detIso_EB_pt);
  MonitorElement *Denominator_detIso_EB_pt = theDbe->get(denpath_detIso_EB_pt);
  
  if (Numerator_detIso_EB_pt && Denominator_detIso_EB_pt){
 
    TH1F *h_numerator_detIso_EB_pt   = Numerator_detIso_EB_pt->getTH1F();
    TH1F *h_denominator_detIso_EB_pt = Denominator_detIso_EB_pt->getTH1F();

    TH1F *h_eff_detIso_EB_pt = h_eff_pt_EB_detIsoTightMu->getTH1F();
    
    if (h_eff_detIso_EB_pt->GetSumw2N() == 0) h_eff_detIso_EB_pt->Sumw2();  
    
    h_eff_detIso_EB_pt->Divide(h_numerator_detIso_EB_pt, h_denominator_detIso_EB_pt, 1., 1., "B");

  }


  /// --- Tight Muon + Detector Isolation  efficiency vs muon Pt [EE] 
  string numpath_detIso_EE_pt = "Muons/EfficiencyAnalyzer/passProbes_EE_detIsoTightMu_pt";
  string denpath_detIso_EE_pt = "Muons/EfficiencyAnalyzer/allProbes_EE_TightMu_pt";
  
  MonitorElement *Numerator_detIso_EE_pt   = theDbe->get(numpath_detIso_EE_pt);
  MonitorElement *Denominator_detIso_EE_pt = theDbe->get(denpath_detIso_EE_pt);
  
  if (Numerator_detIso_EE_pt && Denominator_detIso_EE_pt){
 
    TH1F *h_numerator_detIso_EE_pt   = Numerator_detIso_EE_pt->getTH1F();
    TH1F *h_denominator_detIso_EE_pt = Denominator_detIso_EE_pt->getTH1F();

    TH1F *h_eff_detIso_EE_pt = h_eff_pt_EE_detIsoTightMu->getTH1F();
    
    if (h_eff_detIso_EE_pt->GetSumw2N() == 0) h_eff_detIso_EE_pt->Sumw2();  
    
    h_eff_detIso_EE_pt->Divide(h_numerator_detIso_EE_pt, h_denominator_detIso_EE_pt, 1., 1., "B");

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
    
    if (h_eff_pfIso_pt->GetSumw2N() == 0) h_eff_pfIso_pt->Sumw2();  
    
    h_eff_pfIso_pt->Divide(h_numerator_pfIso_pt, h_denominator_pfIso_pt, 1., 1., "B");
  }


  /// --- Tight Muon + PF Isolation  efficiency vs muon Pt [EB] 

  string numpath_pfIso_EB_pt = "Muons/EfficiencyAnalyzer/passProbes_EB_pfIsoTightMu_pt";
  string denpath_pfIso_EB_pt = "Muons/EfficiencyAnalyzer/allProbes_EB_TightMu_pt";
  
  MonitorElement *Numerator_pfIso_EB_pt   = theDbe->get(numpath_pfIso_EB_pt);
  MonitorElement *Denominator_pfIso_EB_pt = theDbe->get(denpath_pfIso_EB_pt);
  
  if (Numerator_pfIso_EB_pt && Denominator_pfIso_EB_pt){
 
    TH1F *h_numerator_pfIso_EB_pt   = Numerator_pfIso_EB_pt->getTH1F();
    TH1F *h_denominator_pfIso_EB_pt = Denominator_pfIso_EB_pt->getTH1F();

    TH1F *h_eff_pfIso_EB_pt = h_eff_pt_EB_pfIsoTightMu->getTH1F();
    
    if (h_eff_pfIso_EB_pt->GetSumw2N() == 0) h_eff_pfIso_EB_pt->Sumw2();  
    h_eff_pfIso_EB_pt->Divide(h_numerator_pfIso_EB_pt, h_denominator_pfIso_EB_pt, 1., 1., "B");

  }


  /// --- Tight Muon + PF Isolation  efficiency vs muon Pt [EE] 
  string numpath_pfIso_EE_pt = "Muons/EfficiencyAnalyzer/passProbes_EE_pfIsoTightMu_pt";
  string denpath_pfIso_EE_pt = "Muons/EfficiencyAnalyzer/allProbes_EE_TightMu_pt";
  
  MonitorElement *Numerator_pfIso_EE_pt   = theDbe->get(numpath_pfIso_EE_pt);
  MonitorElement *Denominator_pfIso_EE_pt = theDbe->get(denpath_pfIso_EE_pt);
  
  if (Numerator_pfIso_EE_pt && Denominator_pfIso_EE_pt){
 
    TH1F *h_numerator_pfIso_EE_pt   = Numerator_pfIso_EE_pt->getTH1F();
    TH1F *h_denominator_pfIso_EE_pt = Denominator_pfIso_EE_pt->getTH1F();

    TH1F *h_eff_pfIso_EE_pt = h_eff_pt_EE_pfIsoTightMu->getTH1F();
    
    if (h_eff_pfIso_EE_pt->GetSumw2N() == 0) h_eff_pfIso_EE_pt->Sumw2();  
    
    h_eff_pfIso_EE_pt->Divide(h_numerator_pfIso_EE_pt, h_denominator_pfIso_EE_pt, 1., 1., "B");

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
    
    if (h_eff_pfIso_nvtx->GetSumw2N() == 0) h_eff_pfIso_nvtx->Sumw2();  
    
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
    
    if (h_eff_detIso_nvtx->GetSumw2N() == 0) h_eff_detIso_nvtx->Sumw2();  
    
    h_eff_detIso_nvtx->Divide(h_numerator_detIso_nvtx, h_denominator_detIso_nvtx, 1., 1., "B");

  }


  /// --- Tight Muon + detector-Based Isolation  efficiency vs muon number of reco Vertex [EB]
  numpath_detIso_nvtx = "Muons/EfficiencyAnalyzer/passProbes_EB_detIsoTightMu_nVtx";
  denpath_detIso_nvtx = "Muons/EfficiencyAnalyzer/allProbes_EB_TightMu_nVtx";
  
  Numerator_detIso_nvtx   = theDbe->get(numpath_detIso_nvtx);
  Denominator_detIso_nvtx = theDbe->get(denpath_detIso_nvtx);
  
  if (Numerator_detIso_nvtx && Denominator_detIso_nvtx){
    TH1F *h_numerator_detIso_nvtx   = Numerator_detIso_nvtx->getTH1F();
    TH1F *h_denominator_detIso_nvtx = Denominator_detIso_nvtx->getTH1F();
    
    TH1F *h_eff_detIso_nvtx = h_eff_vtx_EB_detIsoTightMu->getTH1F();
    
    if (h_eff_detIso_nvtx->GetSumw2N() == 0) h_eff_detIso_nvtx->Sumw2();  
    h_eff_detIso_nvtx->Divide(h_numerator_detIso_nvtx, h_denominator_detIso_nvtx, 1., 1., "B");
  }
  
  /// --- Tight Muon + detector-Based Isolation  efficiency vs muon number of reco Vertex [EE]
  numpath_detIso_nvtx = "Muons/EfficiencyAnalyzer/passProbes_EE_detIsoTightMu_nVtx";
  denpath_detIso_nvtx = "Muons/EfficiencyAnalyzer/allProbes_EE_TightMu_nVtx";
  
  Numerator_detIso_nvtx   = theDbe->get(numpath_detIso_nvtx);
  Denominator_detIso_nvtx = theDbe->get(denpath_detIso_nvtx);
  
  if (Numerator_detIso_nvtx && Denominator_detIso_nvtx){
 
    TH1F *h_numerator_detIso_nvtx   = Numerator_detIso_nvtx->getTH1F();
    TH1F *h_denominator_detIso_nvtx = Denominator_detIso_nvtx->getTH1F();

    TH1F *h_eff_detIso_nvtx = h_eff_vtx_EE_detIsoTightMu->getTH1F();
    
    if (h_eff_detIso_nvtx->GetSumw2N() == 0) h_eff_detIso_nvtx->Sumw2();  
    
    h_eff_detIso_nvtx->Divide(h_numerator_detIso_nvtx, h_denominator_detIso_nvtx, 1., 1., "B");

  }


  /// --- Tight Muon + PF-Based Isolation  efficiency vs muon number of reco Vertex [EB]
  numpath_pfIso_nvtx = "Muons/EfficiencyAnalyzer/passProbes_EB_pfIsoTightMu_nVtx";
  denpath_pfIso_nvtx = "Muons/EfficiencyAnalyzer/allProbes_EB_TightMu_nVtx";
  
  Numerator_pfIso_nvtx   = theDbe->get(numpath_pfIso_nvtx);
  Denominator_pfIso_nvtx = theDbe->get(denpath_pfIso_nvtx);
  
  if (Numerator_pfIso_nvtx && Denominator_pfIso_nvtx){
 
    TH1F *h_numerator_pfIso_nvtx   = Numerator_pfIso_nvtx->getTH1F();
    TH1F *h_denominator_pfIso_nvtx = Denominator_pfIso_nvtx->getTH1F();

    TH1F *h_eff_pfIso_nvtx = h_eff_vtx_EB_pfIsoTightMu->getTH1F();
    
    if (h_eff_pfIso_nvtx->GetSumw2N() == 0) h_eff_pfIso_nvtx->Sumw2();  
    
    h_eff_pfIso_nvtx->Divide(h_numerator_pfIso_nvtx, h_denominator_pfIso_nvtx, 1., 1., "B");
  }
  
  /// --- Tight Muon + PF-Based Isolation  efficiency vs muon number of reco Vertex [EE]     
  numpath_pfIso_nvtx = "Muons/EfficiencyAnalyzer/passProbes_EE_pfIsoTightMu_nVtx";
  denpath_pfIso_nvtx = "Muons/EfficiencyAnalyzer/allProbes_EE_TightMu_nVtx";
  
  Numerator_pfIso_nvtx   = theDbe->get(numpath_pfIso_nvtx);
  Denominator_pfIso_nvtx = theDbe->get(denpath_pfIso_nvtx);
  
  if (Numerator_pfIso_nvtx && Denominator_pfIso_nvtx){
 
    TH1F *h_numerator_pfIso_nvtx   = Numerator_pfIso_nvtx->getTH1F();
    TH1F *h_denominator_pfIso_nvtx = Denominator_pfIso_nvtx->getTH1F();

    TH1F *h_eff_pfIso_nvtx = h_eff_vtx_EE_pfIsoTightMu->getTH1F();
    
    if (h_eff_pfIso_nvtx->GetSumw2N() == 0) h_eff_pfIso_nvtx->Sumw2();  
    
    h_eff_pfIso_nvtx->Divide(h_numerator_pfIso_nvtx, h_denominator_pfIso_nvtx, 1., 1., "B");

  }



 /// --- Tight Muon + PF IsodBlation corrected by deltaBeta  efficiency vs muon Pt
  string numpath_pfIsodB_pt = "Muons/EfficiencyAnalyzer/passProbes_pfIsodBTightMu_pt";
  string denpath_pfIsodB_pt = "Muons/EfficiencyAnalyzer/allProbes_TightMu_pt";
  
  MonitorElement *Numerator_pfIsodB_pt   = theDbe->get(numpath_pfIsodB_pt);
  MonitorElement *Denominator_pfIsodB_pt = theDbe->get(denpath_pfIsodB_pt);
  
  if (Numerator_pfIsodB_pt && Denominator_pfIsodB_pt){
 
    TH1F *h_numerator_pfIsodB_pt   = Numerator_pfIsodB_pt->getTH1F();
    TH1F *h_denominator_pfIsodB_pt = Denominator_pfIsodB_pt->getTH1F();

    TH1F *h_eff_pfIsodB_pt = h_eff_pt_pfIsodBTightMu->getTH1F();
    
    if (h_eff_pfIsodB_pt->GetSumw2N() == 0) h_eff_pfIsodB_pt->Sumw2();  
    
    h_eff_pfIsodB_pt->Divide(h_numerator_pfIsodB_pt, h_denominator_pfIsodB_pt, 1., 1., "B");
  }


  /// --- Tight Muon + PF IsodBlation corrected by deltaBeta   efficiency vs muon Pt [EB] 

  string numpath_pfIsodB_EB_pt = "Muons/EfficiencyAnalyzer/passProbes_EB_pfIsodBTightMu_pt";
  string denpath_pfIsodB_EB_pt = "Muons/EfficiencyAnalyzer/allProbes_EB_TightMu_pt";
  
  MonitorElement *Numerator_pfIsodB_EB_pt   = theDbe->get(numpath_pfIsodB_EB_pt);
  MonitorElement *Denominator_pfIsodB_EB_pt = theDbe->get(denpath_pfIsodB_EB_pt);
  
  if (Numerator_pfIsodB_EB_pt && Denominator_pfIsodB_EB_pt){
 
    TH1F *h_numerator_pfIsodB_EB_pt   = Numerator_pfIsodB_EB_pt->getTH1F();
    TH1F *h_denominator_pfIsodB_EB_pt = Denominator_pfIsodB_EB_pt->getTH1F();

    TH1F *h_eff_pfIsodB_EB_pt = h_eff_pt_EB_pfIsodBTightMu->getTH1F();
    
    if (h_eff_pfIsodB_EB_pt->GetSumw2N() == 0) h_eff_pfIsodB_EB_pt->Sumw2();  
    h_eff_pfIsodB_EB_pt->Divide(h_numerator_pfIsodB_EB_pt, h_denominator_pfIsodB_EB_pt, 1., 1., "B");

  }


  /// --- Tight Muon + PF IsodBlation corrected by deltaBeta  efficiency vs muon Pt [EE] 
  string numpath_pfIsodB_EE_pt = "Muons/EfficiencyAnalyzer/passProbes_EE_pfIsodBTightMu_pt";
  string denpath_pfIsodB_EE_pt = "Muons/EfficiencyAnalyzer/allProbes_EE_TightMu_pt";
  
  MonitorElement *Numerator_pfIsodB_EE_pt   = theDbe->get(numpath_pfIsodB_EE_pt);
  MonitorElement *Denominator_pfIsodB_EE_pt = theDbe->get(denpath_pfIsodB_EE_pt);
  
  if (Numerator_pfIsodB_EE_pt && Denominator_pfIsodB_EE_pt){
 
    TH1F *h_numerator_pfIsodB_EE_pt   = Numerator_pfIsodB_EE_pt->getTH1F();
    TH1F *h_denominator_pfIsodB_EE_pt = Denominator_pfIsodB_EE_pt->getTH1F();

    TH1F *h_eff_pfIsodB_EE_pt = h_eff_pt_EE_pfIsodBTightMu->getTH1F();
    
    if (h_eff_pfIsodB_EE_pt->GetSumw2N() == 0) h_eff_pfIsodB_EE_pt->Sumw2();  
    
    h_eff_pfIsodB_EE_pt->Divide(h_numerator_pfIsodB_EE_pt, h_denominator_pfIsodB_EE_pt, 1., 1., "B");

  }


 /// --- Tight Muon + PF Isolation corrected by deltaBeta efficiency vs muon number of reco Vertex

  string numpath_pfIsodB_nvtx = "Muons/EfficiencyAnalyzer/passProbes_pfIsodBTightMu_nVtx";
  string denpath_pfIsodB_nvtx = "Muons/EfficiencyAnalyzer/allProbes_TightMu_nVtx";
  
  MonitorElement *Numerator_pfIsodB_nvtx   = theDbe->get(numpath_pfIsodB_nvtx);
  MonitorElement *Denominator_pfIsodB_nvtx = theDbe->get(denpath_pfIsodB_nvtx);
  
  if (Numerator_pfIsodB_nvtx && Denominator_pfIsodB_nvtx){
 
    TH1F *h_numerator_pfIsodB_nvtx   = Numerator_pfIsodB_nvtx->getTH1F();
    TH1F *h_denominator_pfIsodB_nvtx = Denominator_pfIsodB_nvtx->getTH1F();

    TH1F *h_eff_pfIsodB_nvtx = h_eff_vtx_pfIsodBTightMu->getTH1F();
    
    if (h_eff_pfIsodB_nvtx->GetSumw2N() == 0) h_eff_pfIsodB_nvtx->Sumw2();  
    
    h_eff_pfIsodB_nvtx->Divide(h_numerator_pfIsodB_nvtx, h_denominator_pfIsodB_nvtx, 1., 1., "B");

  }


  /// --- Tight Muon + PF-Based Isolation corrected by deltaBeta efficiency vs muon number of reco Vertex [EB]
  numpath_pfIsodB_nvtx = "Muons/EfficiencyAnalyzer/passProbes_EB_pfIsodBTightMu_nVtx";
  denpath_pfIsodB_nvtx = "Muons/EfficiencyAnalyzer/allProbes_EB_TightMu_nVtx";
  
  Numerator_pfIsodB_nvtx   = theDbe->get(numpath_pfIsodB_nvtx);
  Denominator_pfIsodB_nvtx = theDbe->get(denpath_pfIsodB_nvtx);
  
  if (Numerator_pfIsodB_nvtx && Denominator_pfIsodB_nvtx){
 
    TH1F *h_numerator_pfIsodB_nvtx   = Numerator_pfIsodB_nvtx->getTH1F();
    TH1F *h_denominator_pfIsodB_nvtx = Denominator_pfIsodB_nvtx->getTH1F();

    TH1F *h_eff_pfIsodB_nvtx = h_eff_vtx_EB_pfIsodBTightMu->getTH1F();
    
    if (h_eff_pfIsodB_nvtx->GetSumw2N() == 0) h_eff_pfIsodB_nvtx->Sumw2();  
    
    h_eff_pfIsodB_nvtx->Divide(h_numerator_pfIsodB_nvtx, h_denominator_pfIsodB_nvtx, 1., 1., "B");
  }
  

  /// --- Tight Muon + PF-Based Isolation corrected by deltaBeta efficiency vs muon number of reco Vertex [EE]     
  numpath_pfIsodB_nvtx = "Muons/EfficiencyAnalyzer/passProbes_EE_pfIsodBTightMu_nVtx";
  denpath_pfIsodB_nvtx = "Muons/EfficiencyAnalyzer/allProbes_EE_TightMu_nVtx";
  
  Numerator_pfIsodB_nvtx   = theDbe->get(numpath_pfIsodB_nvtx);
  Denominator_pfIsodB_nvtx = theDbe->get(denpath_pfIsodB_nvtx);
  
  if (Numerator_pfIsodB_nvtx && Denominator_pfIsodB_nvtx){
 
    TH1F *h_numerator_pfIsodB_nvtx   = Numerator_pfIsodB_nvtx->getTH1F();
    TH1F *h_denominator_pfIsodB_nvtx = Denominator_pfIsodB_nvtx->getTH1F();

    TH1F *h_eff_pfIsodB_nvtx = h_eff_vtx_EE_pfIsodBTightMu->getTH1F();
    
    if (h_eff_pfIsodB_nvtx->GetSumw2N() == 0) h_eff_pfIsodB_nvtx->Sumw2();  
    
    h_eff_pfIsodB_nvtx->Divide(h_numerator_pfIsodB_nvtx, h_denominator_pfIsodB_nvtx, 1., 1., "B");

  }


}


void EfficiencyPlotter::endJob(){
  LogTrace(metname)<< "[EfficiencyPlotter] endJob called!";
  theDbe->rmdir("Muons/EfficiencyAnalyzer");
}
  

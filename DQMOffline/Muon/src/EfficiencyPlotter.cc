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
  h_eff_pt_TightMu = theDbe->book1D("Eff_pt_TightMu", "TightMu Efficiency vs Pt", ptBin, ptMin, ptMax);
  h_eff_pt_barrel_TightMu = theDbe->book1D("Eff_pt_barrel_TightMu", "Barrel: TightMu Efficiency vs Pt", ptBin, ptMin, ptMax);
  h_eff_pt_endcap_TightMu = theDbe->book1D("Eff_pt_endcap_TightMu", "Endcap: TightMu Efficiency vs Pt", ptBin, ptMin, ptMax);


  h_eff_eta_TightMu->setEfficiencyFlag();
  h_eff_hp_eta_TightMu->setEfficiencyFlag();
  h_eff_phi_TightMu->setEfficiencyFlag();
  h_eff_pt_TightMu->setEfficiencyFlag();
  h_eff_pt_barrel_TightMu->setEfficiencyFlag();
  h_eff_pt_endcap_TightMu->setEfficiencyFlag();

<<<<<<< EfficiencyPlotter.cc
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
=======
>>>>>>> 1.8
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
  
 

  string numpath_pt = "Muons/EfficiencyAnalyzer/passProbes_TightMu_pt";
  string denpath_pt = "Muons/EfficiencyAnalyzer/allProbes_pt";
  
  MonitorElement *Numerator_pt   = theDbe->get(numpath_pt);
  MonitorElement *Denominator_pt = theDbe->get(denpath_pt);
  
  if (Numerator_pt && Denominator_pt){
 
    TH1F *h_numerator_pt   = Numerator_pt->getTH1F();
    TH1F *h_denominator_pt = Denominator_pt->getTH1F();

    TH1F *h_eff_pt = h_eff_pt_TightMu->getTH1F();
    
    if (h_eff_pt->GetSumw2N() == 0) h_eff_pt->Sumw2();
 
    
    h_eff_pt->Divide(h_numerator_pt, h_denominator_pt, 1., 1., "B");

  }

  string numpath_barrel_pt = "Muons/EfficiencyAnalyzer/passProbes_TightMu_barrel_pt";
  string denpath_barrel_pt = "Muons/EfficiencyAnalyzer/allProbes_barrel_pt";
  
  MonitorElement *Numerator_barrel_pt   = theDbe->get(numpath_barrel_pt);
  MonitorElement *Denominator_barrel_pt = theDbe->get(denpath_barrel_pt);
  
  if (Numerator_barrel_pt && Denominator_barrel_pt){
 
    TH1F *h_numerator_barrel_pt   = Numerator_barrel_pt->getTH1F();
    TH1F *h_denominator_barrel_pt = Denominator_barrel_pt->getTH1F();

    TH1F *h_eff_barrel_pt = h_eff_pt_barrel_TightMu->getTH1F();
    
    if (h_eff_barrel_pt->GetSumw2N() == 0) h_eff_barrel_pt->Sumw2();

    h_eff_barrel_pt->Divide(h_numerator_barrel_pt, h_denominator_barrel_pt, 1., 1., "B");

  }

  string numpath_endcap_pt = "Muons/EfficiencyAnalyzer/passProbes_TightMu_endcap_pt";
  string denpath_endcap_pt = "Muons/EfficiencyAnalyzer/allProbes_endcap_pt";
  
  MonitorElement *Numerator_endcap_pt   = theDbe->get(numpath_endcap_pt);
  MonitorElement *Denominator_endcap_pt = theDbe->get(denpath_endcap_pt);
  
  if (Numerator_endcap_pt && Denominator_endcap_pt){
 
    TH1F *h_numerator_endcap_pt   = Numerator_endcap_pt->getTH1F();
    TH1F *h_denominator_endcap_pt = Denominator_endcap_pt->getTH1F();

    TH1F *h_eff_endcap_pt = h_eff_pt_endcap_TightMu->getTH1F();
      
    if (h_eff_endcap_pt->GetSumw2N() == 0) h_eff_endcap_pt->Sumw2();

    h_eff_endcap_pt->Divide(h_numerator_endcap_pt, h_denominator_endcap_pt, 1., 1., "B");

  }

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

    /*   for(int i=1; i<=ptBin; i++){
      if (h_denominator->GetBinContent(i) == 0)  efficiency = 0;
      else                               	 efficiency = double(h_numerator->GetBinContent(i))/double(h_denominator->GetBinContent(i));
      ptEfficiency->setBinContent(i,efficiency);
    }
    }*/
  

  /*
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
      if(h_numerator->GetXaxis()->GetNbins()!=ptBin ||
	h_numerator->GetXaxis()->GetXmax()!=ptBin  ||
	h_numerator->GetXaxis()->GetXmin()!=ptBin){
	LogTrace(metname)<<"[EfficiencyPlotter] wrong histo binning on eta histograms";
	return;
	}
  
    for(int i=1; i<=phiBin; i++){
      if (h_denominator->GetBinContent(i) == 0)  efficiency = 0;
      else                               	 efficiency = double(h_numerator->GetBinContent(i))/double(h_denominator->GetBinContent(i));
      phiEfficiency->setBinContent(i,efficiency);
    }

    }*/
}


void EfficiencyPlotter::endJob(){
  LogTrace(metname)<< "[EfficiencyPlotter] endJob called!";
  theDbe->rmdir("Muons/EfficiencyAnalyzer");
}
  

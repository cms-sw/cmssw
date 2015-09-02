#include "DQMOffline/Muon/interface/EfficiencyPlotter.h"

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
  
  ID_    = parameters.getParameter<string>("MuonID");
}
EfficiencyPlotter::~EfficiencyPlotter(){}

void EfficiencyPlotter::dqmEndJob(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter) {
  
  ibooker.setCurrentFolder("Muons/EfficiencyAnalyzer");
  
  // efficiency plot
  h_eff_eta_ID          = ibooker.book1D("Eff_eta_"+ID_,          ID_+" Eff. vs #eta",                  etaBin, etaMin, etaMax);
  h_eff_hp_eta_ID       = ibooker.book1D("Eff_hp_eta_"+ID_,       "High Pt "+ID_+" Eff. vs #eta",       etaBin, etaMin, etaMax);
  h_eff_phi_ID          = ibooker.book1D("Eff_phi_"+ID_,          ID_+" Eff. vs #phi",                  phiBin, phiMin, phiMax);
  h_eff_pt_ID           = ibooker.book1D("Eff_pt_"+ID_,           ID_+" Eff. vs Pt",                    ptBin, ptMin, ptMax);
  h_eff_pt_EB_ID        = ibooker.book1D("Eff_pt_EB_"+ID_,        "Barrel: "+ID_+" Eff. vs Pt",         ptBin, ptMin, ptMax);
  h_eff_pt_EE_ID        = ibooker.book1D("Eff_pt_EE_"+ID_,        "Endcap: "+ID_+" Eff. vs Pt",         ptBin, ptMin, ptMax);
  h_eff_pt_detIsoID     = ibooker.book1D("Eff_pt_detIso"+ID_,     "detIso"+ID_+" Efficiency vs Pt",     ptBin, ptMin, ptMax);
  h_eff_pt_EB_detIsoID  = ibooker.book1D("Eff_pt_EB_detIso"+ID_,  "Barrel: detIso"+ID_+" Eff. vs Pt",   ptBin, ptMin, ptMax);
  h_eff_pt_EE_detIsoID  = ibooker.book1D("Eff_pt_EE_detIso"+ID_,  "Endcap: detIso"+ID_+" Eff. vs Pt",   ptBin, ptMin, ptMax);
  h_eff_pt_pfIsoID      = ibooker.book1D("Eff_pt_pfIso"+ID_,      "pfIso"+ID_+" Eff. vs Pt",            ptBin, ptMin, ptMax);
  h_eff_pt_EB_pfIsoID   = ibooker.book1D("Eff_pt_EB_pfIso"+ID_,   "Barrel: pfIso"+ID_+" Eff. vs Pt",    ptBin, ptMin, ptMax);
  h_eff_pt_EE_pfIsoID   = ibooker.book1D("Eff_pt_EE_pfIso"+ID_,   "Endcap: pfIso"+ID_+" Eff. vs Pt",    ptBin, ptMin, ptMax);
  h_eff_vtx_detIsoID    = ibooker.book1D("Eff_vtx_detIso"+ID_,    "detIso"+ID_+" Eff. vs nVtx",         vtxBin, vtxMin, vtxMax);
  h_eff_vtx_pfIsoID     = ibooker.book1D("Eff_vtx_pfIso"+ID_,     "pfIso"+ID_+" Eff. vs nVtx",          vtxBin, vtxMin, vtxMax);
  h_eff_vtx_EB_detIsoID = ibooker.book1D("Eff_vtx_EB_detIso"+ID_, "Barrel: detIso"+ID_+" Eff. vs nVtx", vtxBin, vtxMin, vtxMax);
  h_eff_vtx_EB_pfIsoID  = ibooker.book1D("Eff_vtx_EB_pfIso"+ID_,  "Barrel: pfIso"+ID_+" Eff. vs nVtx",  vtxBin, vtxMin, vtxMax);
  h_eff_vtx_EE_detIsoID = ibooker.book1D("Eff_vtx_EE_detIso"+ID_, "Endcap: detIso"+ID_+" Eff. vs nVtx", vtxBin, vtxMin, vtxMax);
  h_eff_vtx_EE_pfIsoID  = ibooker.book1D("Eff_vtx_EE_pfIso"+ID_,  "Endcap: pfIso"+ID_+" Eff. vs nVtx",  vtxBin, vtxMin, vtxMax);

  h_eff_pt_pfIsodBID      = ibooker.book1D("Eff_pt_pfIsodB"+ID_,      "pfIso"+ID_+" (deltaBeta) Eff. vs Pt",            ptBin, ptMin, ptMax);
  h_eff_pt_EB_pfIsodBID   = ibooker.book1D("Eff_pt_EB_pfIsodB"+ID_,   "Barrel: pfIso"+ID_+" (deltaBeta) Eff. vs Pt",    ptBin, ptMin, ptMax);
  h_eff_pt_EE_pfIsodBID   = ibooker.book1D("Eff_pt_EE_pfIsodB"+ID_,   "Endcap: pfIso"+ID_+" (deltaBeta) Eff. vs Pt",    ptBin, ptMin, ptMax);
  h_eff_vtx_pfIsodBID     = ibooker.book1D("Eff_vtx_pfIsodB"+ID_,     "pfIso"+ID_+" (deltaBeta) Eff. vs nVtx",          vtxBin, vtxMin, vtxMax);
  h_eff_vtx_EB_pfIsodBID  = ibooker.book1D("Eff_vtx_EB_pfIsodB"+ID_,  "Barrel: pfIso"+ID_+" (deltaBeta) Eff. vs nVtx",  vtxBin, vtxMin, vtxMax);
  h_eff_vtx_EE_pfIsodBID  = ibooker.book1D("Eff_vtx_EE_pfIsodB"+ID_,  "Endcap: pfIso"+ID_+" (deltaBeta) Eff. vs nVtx",  vtxBin, vtxMin, vtxMax);

  // This prevents this ME to be normalized when drawn into the GUI
  h_eff_eta_ID         ->setEfficiencyFlag();
  h_eff_hp_eta_ID      ->setEfficiencyFlag();
  h_eff_phi_ID         ->setEfficiencyFlag();
  h_eff_pt_ID          ->setEfficiencyFlag();
  h_eff_pt_EB_ID       ->setEfficiencyFlag();
  h_eff_pt_EE_ID       ->setEfficiencyFlag();
  h_eff_pt_detIsoID    ->setEfficiencyFlag();
  h_eff_pt_EB_detIsoID ->setEfficiencyFlag();
  h_eff_pt_EE_detIsoID ->setEfficiencyFlag();
  h_eff_pt_pfIsoID     ->setEfficiencyFlag();
  h_eff_pt_EB_pfIsoID  ->setEfficiencyFlag();
  h_eff_pt_EE_pfIsoID  ->setEfficiencyFlag();
  h_eff_vtx_detIsoID   ->setEfficiencyFlag();
  h_eff_vtx_pfIsoID    ->setEfficiencyFlag();
  h_eff_vtx_EB_detIsoID->setEfficiencyFlag();
  h_eff_vtx_EB_pfIsoID ->setEfficiencyFlag();
  h_eff_vtx_EE_detIsoID->setEfficiencyFlag();
  h_eff_vtx_EE_pfIsoID ->setEfficiencyFlag();

  h_eff_pt_pfIsodBID   ->setEfficiencyFlag();
  h_eff_pt_EB_pfIsodBID ->setEfficiencyFlag();
  h_eff_pt_EE_pfIsodBID ->setEfficiencyFlag();
  h_eff_vtx_pfIsodBID   ->setEfficiencyFlag();
  h_eff_vtx_EB_pfIsodBID ->setEfficiencyFlag();
  h_eff_vtx_EE_pfIsodBID ->setEfficiencyFlag();


  // AXIS TITLES....
  h_eff_hp_eta_ID      ->setAxisTitle("#eta",         1);  
  h_eff_eta_ID         ->setAxisTitle("#eta",         1);  
  h_eff_phi_ID         ->setAxisTitle("#phi",         1);  
  h_eff_pt_ID          ->setAxisTitle("p_{T} (GeV)",  1);  
  h_eff_pt_EB_ID       ->setAxisTitle("p_{T} (GeV)",  1);  
  h_eff_pt_EE_ID       ->setAxisTitle("p_{T} (GeV)",  1);  
  h_eff_pt_detIsoID    ->setAxisTitle("p_{T} (GeV)",  1);  
  h_eff_pt_EB_detIsoID ->setAxisTitle("p_{T} (GeV)",  1);  
  h_eff_pt_EE_detIsoID ->setAxisTitle("p_{T} (GeV)",  1);  
  h_eff_pt_pfIsoID     ->setAxisTitle("p_{T} (GeV)",  1);  
  h_eff_pt_EB_pfIsoID  ->setAxisTitle("p_{T} (GeV)",  1);  
  h_eff_pt_EE_pfIsoID  ->setAxisTitle("p_{T} (GeV)",  1);  
  h_eff_vtx_detIsoID   ->setAxisTitle("Number of PV", 1);
  h_eff_vtx_pfIsoID    ->setAxisTitle("Number of PV", 1);
  h_eff_vtx_EB_detIsoID->setAxisTitle("Number of PV", 1);
  h_eff_vtx_EB_pfIsoID ->setAxisTitle("Number of PV", 1);
  h_eff_vtx_EE_detIsoID->setAxisTitle("Number of PV", 1);
  h_eff_vtx_EE_pfIsoID ->setAxisTitle("Number of PV", 1);

  h_eff_pt_pfIsodBID     ->setAxisTitle("p_{T} (GeV)",  1); 
  h_eff_pt_EB_pfIsodBID  ->setAxisTitle("p_{T} (GeV)",  1);  
  h_eff_pt_EE_pfIsodBID  ->setAxisTitle("p_{T} (GeV)",  1);   
  h_eff_vtx_pfIsodBID    ->setAxisTitle("Number of PV", 1);
  h_eff_vtx_EB_pfIsodBID ->setAxisTitle("Number of PV", 1);
  h_eff_vtx_EE_pfIsodBID ->setAxisTitle("Number of PV", 1);

  
  /// --- Tight Muon efficiency vs muon Pt 
  string inputdir = "Muons/EfficiencyAnalyzer/"+ID_;
  string numpath_pt = inputdir+"/passProbes_ID_pt";
  string denpath_pt = inputdir+"/allProbes_pt";
  
  MonitorElement *Numerator_pt   = igetter.get(numpath_pt);
  MonitorElement *Denominator_pt = igetter.get(denpath_pt);
  
  if (Numerator_pt && Denominator_pt){
    TH1F *h_numerator_pt   = Numerator_pt->getTH1F();
    TH1F *h_denominator_pt = Denominator_pt->getTH1F();
    TH1F *h_eff_pt         = h_eff_pt_ID->getTH1F();
    
    if (h_eff_pt->GetSumw2N() == 0) h_eff_pt->Sumw2();  
    h_eff_pt->Divide(h_numerator_pt, h_denominator_pt, 1., 1., "B");
  }
  
  /// --- Tight Muon efficiency vs muon Pt [EB]
  string numpath_EB_pt = inputdir+"/passProbes_ID_EB_pt";
  string denpath_EB_pt = inputdir+"/allProbes_EB_pt";
  
  MonitorElement *Numerator_EB_pt   = igetter.get(numpath_EB_pt);
  MonitorElement *Denominator_EB_pt = igetter.get(denpath_EB_pt);
  
  if (Numerator_EB_pt && Denominator_EB_pt){
    TH1F *h_numerator_EB_pt   = Numerator_EB_pt->getTH1F();
    TH1F *h_denominator_EB_pt = Denominator_EB_pt->getTH1F();
    TH1F *h_eff_EB_pt         = h_eff_pt_EB_ID->getTH1F();
    
    if (h_eff_EB_pt->GetSumw2N() == 0) h_eff_EB_pt->Sumw2();  
    h_eff_EB_pt->Divide(h_numerator_EB_pt, h_denominator_EB_pt, 1., 1., "B");
  }

  /// --- Tight Muon efficiency vs muon Pt [EE]
  string numpath_EE_pt = inputdir+"/passProbes_ID_EE_pt";
  string denpath_EE_pt = inputdir+"/allProbes_EE_pt";
  
  MonitorElement *Numerator_EE_pt   = igetter.get(numpath_EE_pt);
  MonitorElement *Denominator_EE_pt = igetter.get(denpath_EE_pt);
  
  if (Numerator_EE_pt && Denominator_EE_pt){
    TH1F *h_numerator_EE_pt   = Numerator_EE_pt->getTH1F();
    TH1F *h_denominator_EE_pt = Denominator_EE_pt->getTH1F();
    TH1F *h_eff_EE_pt         = h_eff_pt_EE_ID->getTH1F();
    
    if (h_eff_EE_pt->GetSumw2N() == 0) h_eff_EE_pt->Sumw2();  
    h_eff_EE_pt->Divide(h_numerator_EE_pt, h_denominator_EE_pt, 1., 1., "B");
  }

  /// --- Tight Muon efficiency vs muon Eta
  string numpath_eta = inputdir+"/passProbes_ID_eta";
  string denpath_eta = inputdir+"/allProbes_eta";
  
  MonitorElement *Numerator_eta   = igetter.get(numpath_eta);
  MonitorElement *Denominator_eta = igetter.get(denpath_eta);
  
  if (Numerator_eta && Denominator_eta){
 
    TH1F *h_numerator_eta   = Numerator_eta->getTH1F();
    TH1F *h_denominator_eta = Denominator_eta->getTH1F();

    TH1F *h_eff_eta = h_eff_eta_ID->getTH1F();
    
    if (h_eff_eta->GetSumw2N() == 0) h_eff_eta->Sumw2();  
    
    h_eff_eta->Divide(h_numerator_eta, h_denominator_eta, 1., 1., "B");

  }

  /// --- Tight Muon efficiency vs muon Eta [ pt > 20 ] 

  string numpath_hp_eta = inputdir+"/passProbes_ID_hp_eta";
  string denpath_hp_eta = inputdir+"/allProbes_hp_eta";
  
  MonitorElement *Numerator_hp_eta   = igetter.get(numpath_hp_eta);
  MonitorElement *Denominator_hp_eta = igetter.get(denpath_hp_eta);
  
  if (Numerator_hp_eta && Denominator_hp_eta){
 
    TH1F *h_numerator_hp_eta   = Numerator_hp_eta->getTH1F();
    TH1F *h_denominator_hp_eta = Denominator_hp_eta->getTH1F();

    TH1F *h_eff_hp_eta = h_eff_hp_eta_ID->getTH1F();
    
    if (h_eff_hp_eta->GetSumw2N() == 0) h_eff_hp_eta->Sumw2();  
    
    h_eff_hp_eta->Divide(h_numerator_hp_eta, h_denominator_hp_eta, 1., 1., "B");

  }

  /// --- Tight Muon efficiency vs muon Phi 

  string numpath_phi = inputdir+"/passProbes_ID_phi";
  string denpath_phi = inputdir+"/allProbes_phi";
  
  MonitorElement *Numerator_phi   = igetter.get(numpath_phi);
  MonitorElement *Denominator_phi = igetter.get(denpath_phi);
  
  if (Numerator_phi && Denominator_phi){
 
    TH1F *h_numerator_phi   = Numerator_phi->getTH1F();
    TH1F *h_denominator_phi = Denominator_phi->getTH1F();

    TH1F *h_eff_phi = h_eff_phi_ID->getTH1F();
    
    if (h_eff_phi->GetSumw2N() == 0) h_eff_phi->Sumw2();  
    
    h_eff_phi->Divide(h_numerator_phi, h_denominator_phi, 1., 1., "B");

  }


  /// --- Tight Muon + Detector Isolation  efficiency vs muon Pt
  string numpath_detIso_pt = inputdir+"/passProbes_detIsoID_pt";
  string denpath_detIso_pt = inputdir+"/allProbes_ID_pt";
  
  MonitorElement *Numerator_detIso_pt   = igetter.get(numpath_detIso_pt);
  MonitorElement *Denominator_detIso_pt = igetter.get(denpath_detIso_pt);
  
  if (Numerator_detIso_pt && Denominator_detIso_pt){
 
    TH1F *h_numerator_detIso_pt   = Numerator_detIso_pt->getTH1F();
    TH1F *h_denominator_detIso_pt = Denominator_detIso_pt->getTH1F();

    TH1F *h_eff_detIso_pt = h_eff_pt_detIsoID->getTH1F();
    
    if (h_eff_detIso_pt->GetSumw2N() == 0) h_eff_detIso_pt->Sumw2();  
    h_eff_detIso_pt->Divide(h_numerator_detIso_pt, h_denominator_detIso_pt, 1., 1., "B");
    
  }


  /// --- Tight Muon + Detector Isolation  efficiency vs muon Pt [EB] 
  string numpath_detIso_EB_pt = inputdir+"/passProbes_EB_detIsoID_pt";
  string denpath_detIso_EB_pt = inputdir+"/allProbes_EB_ID_pt";
  
  MonitorElement *Numerator_detIso_EB_pt   = igetter.get(numpath_detIso_EB_pt);
  MonitorElement *Denominator_detIso_EB_pt = igetter.get(denpath_detIso_EB_pt);
  
  if (Numerator_detIso_EB_pt && Denominator_detIso_EB_pt){
 
    TH1F *h_numerator_detIso_EB_pt   = Numerator_detIso_EB_pt->getTH1F();
    TH1F *h_denominator_detIso_EB_pt = Denominator_detIso_EB_pt->getTH1F();

    TH1F *h_eff_detIso_EB_pt = h_eff_pt_EB_detIsoID->getTH1F();
    
    if (h_eff_detIso_EB_pt->GetSumw2N() == 0) h_eff_detIso_EB_pt->Sumw2();  
    
    h_eff_detIso_EB_pt->Divide(h_numerator_detIso_EB_pt, h_denominator_detIso_EB_pt, 1., 1., "B");

  }


  /// --- Tight Muon + Detector Isolation  efficiency vs muon Pt [EE] 
  string numpath_detIso_EE_pt = inputdir+"/passProbes_EE_detIsoID_pt";
  string denpath_detIso_EE_pt = inputdir+"/allProbes_EE_ID_pt";
  
  MonitorElement *Numerator_detIso_EE_pt   = igetter.get(numpath_detIso_EE_pt);
  MonitorElement *Denominator_detIso_EE_pt = igetter.get(denpath_detIso_EE_pt);
  
  if (Numerator_detIso_EE_pt && Denominator_detIso_EE_pt){
 
    TH1F *h_numerator_detIso_EE_pt   = Numerator_detIso_EE_pt->getTH1F();
    TH1F *h_denominator_detIso_EE_pt = Denominator_detIso_EE_pt->getTH1F();

    TH1F *h_eff_detIso_EE_pt = h_eff_pt_EE_detIsoID->getTH1F();
    
    if (h_eff_detIso_EE_pt->GetSumw2N() == 0) h_eff_detIso_EE_pt->Sumw2();  
    
    h_eff_detIso_EE_pt->Divide(h_numerator_detIso_EE_pt, h_denominator_detIso_EE_pt, 1., 1., "B");

  }


 /// --- Tight Muon + PF Isolation  efficiency vs muon Pt
  string numpath_pfIso_pt = inputdir+"/passProbes_pfIsoID_pt";
  string denpath_pfIso_pt = inputdir+"/allProbes_ID_pt";
  
  MonitorElement *Numerator_pfIso_pt   = igetter.get(numpath_pfIso_pt);
  MonitorElement *Denominator_pfIso_pt = igetter.get(denpath_pfIso_pt);
  
  if (Numerator_pfIso_pt && Denominator_pfIso_pt){
 
    TH1F *h_numerator_pfIso_pt   = Numerator_pfIso_pt->getTH1F();
    TH1F *h_denominator_pfIso_pt = Denominator_pfIso_pt->getTH1F();

    TH1F *h_eff_pfIso_pt = h_eff_pt_pfIsoID->getTH1F();
    
    if (h_eff_pfIso_pt->GetSumw2N() == 0) h_eff_pfIso_pt->Sumw2();  
    
    h_eff_pfIso_pt->Divide(h_numerator_pfIso_pt, h_denominator_pfIso_pt, 1., 1., "B");
  }


  /// --- Tight Muon + PF Isolation  efficiency vs muon Pt [EB] 

  string numpath_pfIso_EB_pt = inputdir+"/passProbes_EB_pfIsoID_pt";
  string denpath_pfIso_EB_pt = inputdir+"/allProbes_EB_ID_pt";
  
  MonitorElement *Numerator_pfIso_EB_pt   = igetter.get(numpath_pfIso_EB_pt);
  MonitorElement *Denominator_pfIso_EB_pt = igetter.get(denpath_pfIso_EB_pt);
  
  if (Numerator_pfIso_EB_pt && Denominator_pfIso_EB_pt){
 
    TH1F *h_numerator_pfIso_EB_pt   = Numerator_pfIso_EB_pt->getTH1F();
    TH1F *h_denominator_pfIso_EB_pt = Denominator_pfIso_EB_pt->getTH1F();

    TH1F *h_eff_pfIso_EB_pt = h_eff_pt_EB_pfIsoID->getTH1F();
    
    if (h_eff_pfIso_EB_pt->GetSumw2N() == 0) h_eff_pfIso_EB_pt->Sumw2();  
    h_eff_pfIso_EB_pt->Divide(h_numerator_pfIso_EB_pt, h_denominator_pfIso_EB_pt, 1., 1., "B");

  }


  /// --- Tight Muon + PF Isolation  efficiency vs muon Pt [EE] 
  string numpath_pfIso_EE_pt = inputdir+"/passProbes_EE_pfIsoID_pt";
  string denpath_pfIso_EE_pt = inputdir+"/allProbes_EE_ID_pt";
  
  MonitorElement *Numerator_pfIso_EE_pt   = igetter.get(numpath_pfIso_EE_pt);
  MonitorElement *Denominator_pfIso_EE_pt = igetter.get(denpath_pfIso_EE_pt);
  
  if (Numerator_pfIso_EE_pt && Denominator_pfIso_EE_pt){
 
    TH1F *h_numerator_pfIso_EE_pt   = Numerator_pfIso_EE_pt->getTH1F();
    TH1F *h_denominator_pfIso_EE_pt = Denominator_pfIso_EE_pt->getTH1F();

    TH1F *h_eff_pfIso_EE_pt = h_eff_pt_EE_pfIsoID->getTH1F();
    
    if (h_eff_pfIso_EE_pt->GetSumw2N() == 0) h_eff_pfIso_EE_pt->Sumw2();  
    
    h_eff_pfIso_EE_pt->Divide(h_numerator_pfIso_EE_pt, h_denominator_pfIso_EE_pt, 1., 1., "B");

  }


  /// --- Tight Muon + PF Isolation  efficiency vs muon number of reco Vertex

  string numpath_pfIso_nvtx = inputdir+"/passProbes_pfIsoID_nVtx";
  string denpath_pfIso_nvtx = inputdir+"/allProbes_ID_nVtx";
  
  MonitorElement *Numerator_pfIso_nvtx   = igetter.get(numpath_pfIso_nvtx);
  MonitorElement *Denominator_pfIso_nvtx = igetter.get(denpath_pfIso_nvtx);
  
  if (Numerator_pfIso_nvtx && Denominator_pfIso_nvtx){
 
    TH1F *h_numerator_pfIso_nvtx   = Numerator_pfIso_nvtx->getTH1F();
    TH1F *h_denominator_pfIso_nvtx = Denominator_pfIso_nvtx->getTH1F();

    TH1F *h_eff_pfIso_nvtx = h_eff_vtx_pfIsoID->getTH1F();
    
    if (h_eff_pfIso_nvtx->GetSumw2N() == 0) h_eff_pfIso_nvtx->Sumw2();  
    
    h_eff_pfIso_nvtx->Divide(h_numerator_pfIso_nvtx, h_denominator_pfIso_nvtx, 1., 1., "B");

  }



  /// --- Tight Muon + detector-Based Isolation  efficiency vs muon number of reco Vertex
  string numpath_detIso_nvtx = inputdir+"/passProbes_detIsoID_nVtx";
  string denpath_detIso_nvtx = inputdir+"/allProbes_ID_nVtx";
  
  MonitorElement *Numerator_detIso_nvtx   = igetter.get(numpath_detIso_nvtx);
  MonitorElement *Denominator_detIso_nvtx = igetter.get(denpath_detIso_nvtx);
  
  if (Numerator_detIso_nvtx && Denominator_detIso_nvtx){
 
    TH1F *h_numerator_detIso_nvtx   = Numerator_detIso_nvtx->getTH1F();
    TH1F *h_denominator_detIso_nvtx = Denominator_detIso_nvtx->getTH1F();

    TH1F *h_eff_detIso_nvtx = h_eff_vtx_detIsoID->getTH1F();
    
    if (h_eff_detIso_nvtx->GetSumw2N() == 0) h_eff_detIso_nvtx->Sumw2();  
    
    h_eff_detIso_nvtx->Divide(h_numerator_detIso_nvtx, h_denominator_detIso_nvtx, 1., 1., "B");

  }


  /// --- Tight Muon + detector-Based Isolation  efficiency vs muon number of reco Vertex [EB]
  numpath_detIso_nvtx = inputdir+"/passProbes_EB_detIsoID_nVtx";
  denpath_detIso_nvtx = inputdir+"/allProbes_EB_ID_nVtx";
  
  Numerator_detIso_nvtx   = igetter.get(numpath_detIso_nvtx);
  Denominator_detIso_nvtx = igetter.get(denpath_detIso_nvtx);
  
  if (Numerator_detIso_nvtx && Denominator_detIso_nvtx){
    TH1F *h_numerator_detIso_nvtx   = Numerator_detIso_nvtx->getTH1F();
    TH1F *h_denominator_detIso_nvtx = Denominator_detIso_nvtx->getTH1F();
    
    TH1F *h_eff_detIso_nvtx = h_eff_vtx_EB_detIsoID->getTH1F();
    
    if (h_eff_detIso_nvtx->GetSumw2N() == 0) h_eff_detIso_nvtx->Sumw2();  
    h_eff_detIso_nvtx->Divide(h_numerator_detIso_nvtx, h_denominator_detIso_nvtx, 1., 1., "B");
  }
  
  /// --- Tight Muon + detector-Based Isolation  efficiency vs muon number of reco Vertex [EE]
  numpath_detIso_nvtx = inputdir+"/passProbes_EE_detIsoID_nVtx";
  denpath_detIso_nvtx = inputdir+"/allProbes_EE_ID_nVtx";
  
  Numerator_detIso_nvtx   = igetter.get(numpath_detIso_nvtx);
  Denominator_detIso_nvtx = igetter.get(denpath_detIso_nvtx);
  
  if (Numerator_detIso_nvtx && Denominator_detIso_nvtx){
 
    TH1F *h_numerator_detIso_nvtx   = Numerator_detIso_nvtx->getTH1F();
    TH1F *h_denominator_detIso_nvtx = Denominator_detIso_nvtx->getTH1F();

    TH1F *h_eff_detIso_nvtx = h_eff_vtx_EE_detIsoID->getTH1F();
    
    if (h_eff_detIso_nvtx->GetSumw2N() == 0) h_eff_detIso_nvtx->Sumw2();  
    
    h_eff_detIso_nvtx->Divide(h_numerator_detIso_nvtx, h_denominator_detIso_nvtx, 1., 1., "B");

  }


  /// --- Tight Muon + PF-Based Isolation  efficiency vs muon number of reco Vertex [EB]
  numpath_pfIso_nvtx = inputdir+"/passProbes_EB_pfIsoID_nVtx";
  denpath_pfIso_nvtx = inputdir+"/allProbes_EB_ID_nVtx";
  
  Numerator_pfIso_nvtx   = igetter.get(numpath_pfIso_nvtx);
  Denominator_pfIso_nvtx = igetter.get(denpath_pfIso_nvtx);
  
  if (Numerator_pfIso_nvtx && Denominator_pfIso_nvtx){
 
    TH1F *h_numerator_pfIso_nvtx   = Numerator_pfIso_nvtx->getTH1F();
    TH1F *h_denominator_pfIso_nvtx = Denominator_pfIso_nvtx->getTH1F();

    TH1F *h_eff_pfIso_nvtx = h_eff_vtx_EB_pfIsoID->getTH1F();
    
    if (h_eff_pfIso_nvtx->GetSumw2N() == 0) h_eff_pfIso_nvtx->Sumw2();  
    
    h_eff_pfIso_nvtx->Divide(h_numerator_pfIso_nvtx, h_denominator_pfIso_nvtx, 1., 1., "B");
  }
  
  /// --- Tight Muon + PF-Based Isolation  efficiency vs muon number of reco Vertex [EE]     
  numpath_pfIso_nvtx = inputdir+"/passProbes_EE_pfIsoID_nVtx";
  denpath_pfIso_nvtx = inputdir+"/allProbes_EE_ID_nVtx";
  
  Numerator_pfIso_nvtx   = igetter.get(numpath_pfIso_nvtx);
  Denominator_pfIso_nvtx = igetter.get(denpath_pfIso_nvtx);
  
  if (Numerator_pfIso_nvtx && Denominator_pfIso_nvtx){
 
    TH1F *h_numerator_pfIso_nvtx   = Numerator_pfIso_nvtx->getTH1F();
    TH1F *h_denominator_pfIso_nvtx = Denominator_pfIso_nvtx->getTH1F();

    TH1F *h_eff_pfIso_nvtx = h_eff_vtx_EE_pfIsoID->getTH1F();
    
    if (h_eff_pfIso_nvtx->GetSumw2N() == 0) h_eff_pfIso_nvtx->Sumw2();  
    
    h_eff_pfIso_nvtx->Divide(h_numerator_pfIso_nvtx, h_denominator_pfIso_nvtx, 1., 1., "B");

  }



 /// --- Tight Muon + PF IsodBlation corrected by deltaBeta  efficiency vs muon Pt
  string numpath_pfIsodB_pt = inputdir+"/passProbes_pfIsodBID_pt";
  string denpath_pfIsodB_pt = inputdir+"/allProbes_ID_pt";
  
  MonitorElement *Numerator_pfIsodB_pt   = igetter.get(numpath_pfIsodB_pt);
  MonitorElement *Denominator_pfIsodB_pt = igetter.get(denpath_pfIsodB_pt);
  
  if (Numerator_pfIsodB_pt && Denominator_pfIsodB_pt){
 
    TH1F *h_numerator_pfIsodB_pt   = Numerator_pfIsodB_pt->getTH1F();
    TH1F *h_denominator_pfIsodB_pt = Denominator_pfIsodB_pt->getTH1F();

    TH1F *h_eff_pfIsodB_pt = h_eff_pt_pfIsodBID->getTH1F();
    
    if (h_eff_pfIsodB_pt->GetSumw2N() == 0) h_eff_pfIsodB_pt->Sumw2();  
    
    h_eff_pfIsodB_pt->Divide(h_numerator_pfIsodB_pt, h_denominator_pfIsodB_pt, 1., 1., "B");
  }


  /// --- Tight Muon + PF IsodBlation corrected by deltaBeta   efficiency vs muon Pt [EB] 

  string numpath_pfIsodB_EB_pt = inputdir+"/passProbes_EB_pfIsodBID_pt";
  string denpath_pfIsodB_EB_pt = inputdir+"/allProbes_EB_ID_pt";
  
  MonitorElement *Numerator_pfIsodB_EB_pt   = igetter.get(numpath_pfIsodB_EB_pt);
  MonitorElement *Denominator_pfIsodB_EB_pt = igetter.get(denpath_pfIsodB_EB_pt);
  
  if (Numerator_pfIsodB_EB_pt && Denominator_pfIsodB_EB_pt){
 
    TH1F *h_numerator_pfIsodB_EB_pt   = Numerator_pfIsodB_EB_pt->getTH1F();
    TH1F *h_denominator_pfIsodB_EB_pt = Denominator_pfIsodB_EB_pt->getTH1F();

    TH1F *h_eff_pfIsodB_EB_pt = h_eff_pt_EB_pfIsodBID->getTH1F();
    
    if (h_eff_pfIsodB_EB_pt->GetSumw2N() == 0) h_eff_pfIsodB_EB_pt->Sumw2();  
    h_eff_pfIsodB_EB_pt->Divide(h_numerator_pfIsodB_EB_pt, h_denominator_pfIsodB_EB_pt, 1., 1., "B");

  }


  /// --- Tight Muon + PF IsodBlation corrected by deltaBeta  efficiency vs muon Pt [EE] 
  string numpath_pfIsodB_EE_pt = inputdir+"/passProbes_EE_pfIsodBID_pt";
  string denpath_pfIsodB_EE_pt = inputdir+"/allProbes_EE_ID_pt";
  
  MonitorElement *Numerator_pfIsodB_EE_pt   = igetter.get(numpath_pfIsodB_EE_pt);
  MonitorElement *Denominator_pfIsodB_EE_pt = igetter.get(denpath_pfIsodB_EE_pt);
  
  if (Numerator_pfIsodB_EE_pt && Denominator_pfIsodB_EE_pt){
 
    TH1F *h_numerator_pfIsodB_EE_pt   = Numerator_pfIsodB_EE_pt->getTH1F();
    TH1F *h_denominator_pfIsodB_EE_pt = Denominator_pfIsodB_EE_pt->getTH1F();

    TH1F *h_eff_pfIsodB_EE_pt = h_eff_pt_EE_pfIsodBID->getTH1F();
    
    if (h_eff_pfIsodB_EE_pt->GetSumw2N() == 0) h_eff_pfIsodB_EE_pt->Sumw2();  
    
    h_eff_pfIsodB_EE_pt->Divide(h_numerator_pfIsodB_EE_pt, h_denominator_pfIsodB_EE_pt, 1., 1., "B");

  }


 /// --- Tight Muon + PF Isolation corrected by deltaBeta efficiency vs muon number of reco Vertex

  string numpath_pfIsodB_nvtx = inputdir+"/passProbes_pfIsodBID_nVtx";
  string denpath_pfIsodB_nvtx = inputdir+"/allProbes_ID_nVtx";
  
  MonitorElement *Numerator_pfIsodB_nvtx   = igetter.get(numpath_pfIsodB_nvtx);
  MonitorElement *Denominator_pfIsodB_nvtx = igetter.get(denpath_pfIsodB_nvtx);
  
  if (Numerator_pfIsodB_nvtx && Denominator_pfIsodB_nvtx){
 
    TH1F *h_numerator_pfIsodB_nvtx   = Numerator_pfIsodB_nvtx->getTH1F();
    TH1F *h_denominator_pfIsodB_nvtx = Denominator_pfIsodB_nvtx->getTH1F();

    TH1F *h_eff_pfIsodB_nvtx = h_eff_vtx_pfIsodBID->getTH1F();
    
    if (h_eff_pfIsodB_nvtx->GetSumw2N() == 0) h_eff_pfIsodB_nvtx->Sumw2();  
    
    h_eff_pfIsodB_nvtx->Divide(h_numerator_pfIsodB_nvtx, h_denominator_pfIsodB_nvtx, 1., 1., "B");

  }


  /// --- Tight Muon + PF-Based Isolation corrected by deltaBeta efficiency vs muon number of reco Vertex [EB]
  numpath_pfIsodB_nvtx = inputdir+"/passProbes_EB_pfIsodBID_nVtx";
  denpath_pfIsodB_nvtx = inputdir+"/allProbes_EB_ID_nVtx";
  
  Numerator_pfIsodB_nvtx   = igetter.get(numpath_pfIsodB_nvtx);
  Denominator_pfIsodB_nvtx = igetter.get(denpath_pfIsodB_nvtx);
  
  if (Numerator_pfIsodB_nvtx && Denominator_pfIsodB_nvtx){
 
    TH1F *h_numerator_pfIsodB_nvtx   = Numerator_pfIsodB_nvtx->getTH1F();
    TH1F *h_denominator_pfIsodB_nvtx = Denominator_pfIsodB_nvtx->getTH1F();

    TH1F *h_eff_pfIsodB_nvtx = h_eff_vtx_EB_pfIsodBID->getTH1F();
    
    if (h_eff_pfIsodB_nvtx->GetSumw2N() == 0) h_eff_pfIsodB_nvtx->Sumw2();  
    
    h_eff_pfIsodB_nvtx->Divide(h_numerator_pfIsodB_nvtx, h_denominator_pfIsodB_nvtx, 1., 1., "B");
  }
  

  /// --- Tight Muon + PF-Based Isolation corrected by deltaBeta efficiency vs muon number of reco Vertex [EE]     
  numpath_pfIsodB_nvtx = inputdir+"/passProbes_EE_pfIsodBID_nVtx";
  denpath_pfIsodB_nvtx = inputdir+"/allProbes_EE_ID_nVtx";
  
  Numerator_pfIsodB_nvtx   = igetter.get(numpath_pfIsodB_nvtx);
  Denominator_pfIsodB_nvtx = igetter.get(denpath_pfIsodB_nvtx);
  
  if (Numerator_pfIsodB_nvtx && Denominator_pfIsodB_nvtx){
 
    TH1F *h_numerator_pfIsodB_nvtx   = Numerator_pfIsodB_nvtx->getTH1F();
    TH1F *h_denominator_pfIsodB_nvtx = Denominator_pfIsodB_nvtx->getTH1F();

    TH1F *h_eff_pfIsodB_nvtx = h_eff_vtx_EE_pfIsodBID->getTH1F();
    
    if (h_eff_pfIsodB_nvtx->GetSumw2N() == 0) h_eff_pfIsodB_nvtx->Sumw2();  
    
    h_eff_pfIsodB_nvtx->Divide(h_numerator_pfIsodB_nvtx, h_denominator_pfIsodB_nvtx, 1., 1., "B");

  }


}
  

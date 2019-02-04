#include "DQMOffline/Muon/interface/TriggerMatchEfficiencyPlotter.h"
/** \class TriggerMatch monitor
 *  *  *
 *   *   *  DQM monitoring source for Trigger matching efficiency plotter  feature added to miniAOD
 *    *    *
 *     *     *  \author Bibhuprasad Mahakud (Purdue University, West Lafayette, USA)
 *      *      */


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
#include <cstdio>
#include <string>
#include <cmath>
#include "TF1.h"
#include "TH1F.h"

using namespace edm;
using namespace std;

//#define DEBUG

TriggerMatchEfficiencyPlotter::TriggerMatchEfficiencyPlotter(const edm::ParameterSet& ps){
#ifdef DEBUG
  cout << "TriggerMatchEfficiencyPlotter(): Constructor " << endl;
#endif
  parameters = ps;

  triggerhistName1_ = parameters.getParameter<string>("triggerhistName1");
  triggerhistName2_ = parameters.getParameter<string>("triggerhistName2"); 
  theFolder = parameters.getParameter<string>("folder");
}
TriggerMatchEfficiencyPlotter::~TriggerMatchEfficiencyPlotter(){}

void TriggerMatchEfficiencyPlotter::dqmEndJob(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter) {
  
  ibooker.setCurrentFolder(theFolder);
  
  // efficiency plot
  h_eff_Path1_eta_tight      = ibooker.book1D("matchingEff_eta_"+triggerhistName1_+"_tight",triggerhistName1_+":matching Eff. vs #eta",8, -2.5, 2.5);
  h_eff_Path1_pt_tight      = ibooker.book1D("matchingEff_pt_"+triggerhistName1_+"_tight",triggerhistName1_+":matching Eff. vs pt",10, 20, 220);
  h_eff_Path1_phi_tight      = ibooker.book1D("matchingEff_phi_"+triggerhistName1_+"_tight",triggerhistName1_+":matching Eff. vs #phi", 8, -3.0, 3.0);
  h_eff_Path2_eta_tight      = ibooker.book1D("matchingEff_eta_"+triggerhistName2_+"_tight",triggerhistName2_+":matching Eff. vs #eta",8, -2.5, 2.5);
  h_eff_Path2_pt_tight      = ibooker.book1D("matchingEff_pt_"+triggerhistName2_+"_tight",triggerhistName2_+":matching Eff. vs pt",10, 20, 220);
  h_eff_Path2_phi_tight      = ibooker.book1D("matchingEff_phi_"+triggerhistName2_+"_tight",triggerhistName2_+":matching Eff. vs #phi", 8, -3.0, 3.0);



  // This prevents this ME to be normalized when drawn into the GUI
  h_eff_Path1_eta_tight->setEfficiencyFlag();
  h_eff_Path1_pt_tight->setEfficiencyFlag();
  h_eff_Path1_phi_tight->setEfficiencyFlag();
  h_eff_Path2_eta_tight->setEfficiencyFlag();
  h_eff_Path2_pt_tight->setEfficiencyFlag();
  h_eff_Path2_phi_tight->setEfficiencyFlag();

  // AXIS TITLES....
  h_eff_Path1_eta_tight      ->setAxisTitle("#eta",         1);  
  h_eff_Path1_pt_tight       ->setAxisTitle("pt",         1);
  h_eff_Path1_phi_tight       ->setAxisTitle("#phi",         1);  
  h_eff_Path2_eta_tight      ->setAxisTitle("#eta",         1);
  h_eff_Path2_pt_tight       ->setAxisTitle("pt",         1);
  h_eff_Path2_phi_tight       ->setAxisTitle("#phi",         1);



  /// --- Tight Muon trigger mathcing efficiency  
  string inputdir = "Muons_miniAOD/TriggerMatchMonitor/EfficiencyInput";
  string numpath_eta_path1 = inputdir+"/passHLT"+triggerhistName1_+"_eta_Tight";
  string denpath_eta_path1 = inputdir+"/totalHLT"+triggerhistName1_+"_eta_Tight";

  string numpath_pt_path1 = inputdir+"/passHLT"+triggerhistName1_+"_pt_Tight";
  string denpath_pt_path1 = inputdir+"/totalHLT"+triggerhistName1_+"_pt_Tight";
  
  string numpath_phi_path1 = inputdir+"/passHLT"+triggerhistName1_+"_phi_Tight";
  string denpath_phi_path1 = inputdir+"/totalHLT"+triggerhistName1_+"_phi_Tight"; 

  string numpath_eta_path2 = inputdir+"/passHLT"+triggerhistName2_+"_eta_Tight";
  string denpath_eta_path2 = inputdir+"/totalHLT"+triggerhistName2_+"_eta_Tight";

  string numpath_pt_path2 = inputdir+"/passHLT"+triggerhistName2_+"_pt_Tight";
  string denpath_pt_path2 = inputdir+"/totalHLT"+triggerhistName2_+"_pt_Tight";

  string numpath_phi_path2 = inputdir+"/passHLT"+triggerhistName2_+"_phi_Tight";
  string denpath_phi_path2 = inputdir+"/totalHLT"+triggerhistName2_+"_phi_Tight";
  
  MonitorElement *Numerator_eta_path1   = igetter.get(numpath_eta_path1);
  MonitorElement *Denominator_eta_path1 = igetter.get(denpath_eta_path1);

  MonitorElement *Numerator_pt_path1   = igetter.get(numpath_pt_path1);
  MonitorElement *Denominator_pt_path1 = igetter.get(denpath_pt_path1);

  MonitorElement *Numerator_phi_path1   = igetter.get(numpath_phi_path1);
  MonitorElement *Denominator_phi_path1 = igetter.get(denpath_phi_path1);

  MonitorElement *Numerator_eta_path2   = igetter.get(numpath_eta_path2);
  MonitorElement *Denominator_eta_path2 = igetter.get(denpath_eta_path2);

  MonitorElement *Numerator_pt_path2   = igetter.get(numpath_pt_path2);
  MonitorElement *Denominator_pt_path2 = igetter.get(denpath_pt_path2);

  MonitorElement *Numerator_phi_path2   = igetter.get(numpath_phi_path2);
  MonitorElement *Denominator_phi_path2 = igetter.get(denpath_phi_path2); 

  if (Numerator_eta_path1 && Denominator_eta_path1){
    TH1F *h_numerator_eta_path1   = Numerator_eta_path1->getTH1F();
    TH1F *h_denominator_eta_path1 = Denominator_eta_path1->getTH1F();
    TH1F *h_eff_eta_path1         = h_eff_Path1_eta_tight->getTH1F();
    
    if (h_eff_eta_path1->GetSumw2N() == 0) h_eff_eta_path1->Sumw2();  
    h_eff_eta_path1->Divide(h_numerator_eta_path1, h_denominator_eta_path1, 1., 1., "B");
  }

  if (Numerator_pt_path1 && Denominator_pt_path1){
    TH1F *h_numerator_pt_path1   = Numerator_pt_path1->getTH1F();
    TH1F *h_denominator_pt_path1 = Denominator_pt_path1->getTH1F();
    TH1F *h_eff_pt_path1         = h_eff_Path1_pt_tight->getTH1F();

    if (h_eff_pt_path1->GetSumw2N() == 0) h_eff_pt_path1->Sumw2();
    h_eff_pt_path1->Divide(h_numerator_pt_path1, h_denominator_pt_path1, 1., 1., "B");
  }

  if (Numerator_phi_path1 && Denominator_phi_path1){
    TH1F *h_numerator_phi_path1   = Numerator_phi_path1->getTH1F();
    TH1F *h_denominator_phi_path1 = Denominator_phi_path1->getTH1F();
    TH1F *h_eff_phi_path1         = h_eff_Path1_phi_tight->getTH1F();

    if (h_eff_phi_path1->GetSumw2N() == 0) h_eff_phi_path1->Sumw2();
    h_eff_phi_path1->Divide(h_numerator_phi_path1, h_denominator_phi_path1, 1., 1., "B");
  }
 
  //trigger path2
  if (Numerator_eta_path2 && Denominator_eta_path2){
    TH1F *h_numerator_eta_path2   = Numerator_eta_path2->getTH1F();
    TH1F *h_denominator_eta_path2 = Denominator_eta_path2->getTH1F();
    TH1F *h_eff_eta_path2         = h_eff_Path2_eta_tight->getTH1F();

    if (h_eff_eta_path2->GetSumw2N() == 0) h_eff_eta_path2->Sumw2();
    h_eff_eta_path2->Divide(h_numerator_eta_path2, h_denominator_eta_path2, 1., 1., "B");
  }

  if (Numerator_pt_path2 && Denominator_pt_path2){
    TH1F *h_numerator_pt_path2   = Numerator_pt_path2->getTH1F();
    TH1F *h_denominator_pt_path2 = Denominator_pt_path2->getTH1F();
    TH1F *h_eff_pt_path2         = h_eff_Path2_pt_tight->getTH1F();

    if (h_eff_pt_path2->GetSumw2N() == 0) h_eff_pt_path2->Sumw2();
    h_eff_pt_path2->Divide(h_numerator_pt_path2, h_denominator_pt_path2, 1., 1., "B");
  }

  if (Numerator_phi_path2 && Denominator_phi_path2){
    TH1F *h_numerator_phi_path2   = Numerator_phi_path2->getTH1F();
    TH1F *h_denominator_phi_path2 = Denominator_phi_path2->getTH1F();
    TH1F *h_eff_phi_path2         = h_eff_Path2_phi_tight->getTH1F();

    if (h_eff_phi_path2->GetSumw2N() == 0) h_eff_phi_path2->Sumw2();
    h_eff_phi_path2->Divide(h_numerator_phi_path2, h_denominator_phi_path2, 1., 1., "B");
  }


}
  

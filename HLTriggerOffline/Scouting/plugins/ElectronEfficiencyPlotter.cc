#include "ElectronEfficiencyPlotter.h"

// Framework
#include <FWCore/Framework/interface/Event.h>
#include "DataFormats/Common/interface/Handle.h"
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include "DQMServices/Core/interface/DQMStore.h"
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

ElectronEfficiencyPlotter::ElectronEfficiencyPlotter(const edm::ParameterSet &ps) {
  parameters = ps;

  ptBin = parameters.getParameter<int>("ptBin");
  ptMin = parameters.getParameter<double>("ptMin");
  ptMax = parameters.getParameter<double>("ptMax");

  ID_ = parameters.getParameter<string>("sctElectronID");
  theFolder_ = parameters.getParameter<string>("folder");
  sourceFolder_ = parameters.getParameter<string>("srcFolder");
}

ElectronEfficiencyPlotter::~ElectronEfficiencyPlotter() {}

void ElectronEfficiencyPlotter::dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {
  ibooker.setCurrentFolder(theFolder_);

  h_eff_pt_EB_ID = ibooker.book1D("Eff_pt_barrel_" + ID_, ID_ + "Eff. vs Pt (barrel)", ptBin, ptMin, ptMax);
  h_eff_pt_EE_ID = ibooker.book1D("Eff_pt_endcap_" + ID_, ID_ + "Eff. vs Pt (endcap)", ptBin, ptMin, ptMax);
  h_eff_pt_EB_HLT = ibooker.book1D("Eff_pt_barrel_DSTdoubleEG", "DSTdoubleEG Eff. vs Pt (barrel)", ptBin, ptMin, ptMax);
  h_eff_pt_EE_HLT = ibooker.book1D("Eff_pt_endcap_DSTdoubleEG", "DSTdoubleEG Eff. vs Pt (endcap)", ptBin, ptMin, ptMax);

  // Prevent the ME to be normalized when drawn into the GUI
  // h_eff_pt_EB_ID->setEfficiencyFlag();
  // h_eff_pt_EE_ID->setEfficiencyFlag();

  // Axis title
  h_eff_pt_EB_ID->setAxisTitle("p_{T} (GeV)", 1);
  h_eff_pt_EE_ID->setAxisTitle("p_{T} (GeV)", 1);
  h_eff_pt_EB_HLT->setAxisTitle("p_{T} (GeV)", 1);
  h_eff_pt_EE_HLT->setAxisTitle("p_{T} (GeV)", 1);

  MonitorElement *Numerator_pt_barrel = igetter.get(sourceFolder_ + "/resonanceAll_Probe_sctElectron_Pt_Barrel_passID");
  MonitorElement *Numerator_pt_endcap = igetter.get(sourceFolder_ + "/resonanceAll_Probe_sctElectron_Pt_Endcap_passID");
  MonitorElement *Numerator_pt_barrel_hlt =
      igetter.get(sourceFolder_ + "/resonanceAll_Probe_sctElectron_Pt_Barrel_passDSTdoubleEG");
  MonitorElement *Numerator_pt_endcap_hlt =
      igetter.get(sourceFolder_ + "/resonanceAll_Probe_sctElectron_Pt_Endcap_passDSTdoubleEG");
  MonitorElement *Denominator_pt_barrel = igetter.get(sourceFolder_ + "/resonanceAll_Probe_sctElectron_Pt_Barrel");
  MonitorElement *Denominator_pt_endcap = igetter.get(sourceFolder_ + "/resonanceAll_Probe_sctElectron_Pt_Endcap");

  if (Numerator_pt_barrel && Denominator_pt_barrel)
    GetEfficiency(Numerator_pt_barrel, Denominator_pt_barrel, h_eff_pt_EB_ID);
  if (Numerator_pt_endcap && Denominator_pt_endcap)
    GetEfficiency(Numerator_pt_endcap, Denominator_pt_endcap, h_eff_pt_EE_ID);
  if (Numerator_pt_barrel_hlt && Denominator_pt_barrel)
    GetEfficiency(Numerator_pt_barrel_hlt, Denominator_pt_barrel, h_eff_pt_EB_HLT);
  if (Numerator_pt_endcap_hlt && Denominator_pt_endcap)
    GetEfficiency(Numerator_pt_endcap_hlt, Denominator_pt_endcap, h_eff_pt_EE_HLT);
}

void ElectronEfficiencyPlotter::GetEfficiency(MonitorElement *Numerator,
                                              MonitorElement *Denominator,
                                              MonitorElement *Efficiency) {
  TH1F *h_numerator_pt = Numerator->getTH1F();
  TH1F *h_denominator_pt = Denominator->getTH1F();
  TH1F *h_eff_pt = Efficiency->getTH1F();
  if (h_eff_pt->GetSumw2N() == 0)
    h_eff_pt->Sumw2();

  // ReBin
  int nBins = h_eff_pt->GetNbinsX();
  double *binEdges = new double[nBins + 1];
  for (int i = 0; i <= nBins; i++)
    binEdges[i] = h_eff_pt->GetBinLowEdge(i + 1);

  TH1F *h_numerator_pt_rebin = (TH1F *)h_numerator_pt->Rebin(nBins, "num_pt_rebinned", binEdges);
  TH1F *h_denominator_pt_rebin = (TH1F *)h_denominator_pt->Rebin(nBins, "num_pt_rebinned", binEdges);
  h_eff_pt->Divide(h_numerator_pt_rebin, h_denominator_pt_rebin, 1., 1., "B");
}

DEFINE_FWK_MODULE(ElectronEfficiencyPlotter);

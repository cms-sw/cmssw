// system includes
#include <cmath>
#include <string>

// user includes
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// ROOT includes
#include "TF1.h"
#include "TH1F.h"

class ElectronEfficiencyPlotter : public DQMEDHarvester {
public:
  // Constructor
  ElectronEfficiencyPlotter(const edm::ParameterSet &ps);
  // Destructor
  ~ElectronEfficiencyPlotter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

protected:
  // DQM Client Diagnostic
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;

private:
  // counters
  const int ptBin_;
  const double ptMin_;
  const double ptMax_;
  const std::string ID_;
  const std::string theFolder_;
  const std::string sourceFolder_;

  MonitorElement *h_eff_pt_EB_doubleEG_HLT;
  MonitorElement *h_eff_pt_EE_doubleEG_HLT;
  MonitorElement *h_eff_pt_EB_singlePhoton_HLT;
  MonitorElement *h_eff_pt_EE_singlePhoton_HLT;

  void calculateEfficiency(MonitorElement *Numerator, MonitorElement *Denominator, MonitorElement *Efficiency);
};

using namespace edm;
using namespace std;

// Constructor
ElectronEfficiencyPlotter::ElectronEfficiencyPlotter(const edm::ParameterSet &ps)
    : ptBin_{ps.getParameter<int>("ptBin")},
      ptMin_{ps.getParameter<double>("ptMin")},
      ptMax_{ps.getParameter<double>("ptMax")},
      ID_{ps.getParameter<string>("sctElectronID")},
      theFolder_{ps.getParameter<string>("folder")},
      sourceFolder_{ps.getParameter<string>("srcFolder")} {}

void ElectronEfficiencyPlotter::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<int>("ptBin", 5);
  desc.add<double>("ptMin", 0);
  desc.add<double>("ptMax", 100);
  desc.add<string>("sctElectronID", {});
  desc.add<string>("folder", {});
  desc.add<string>("srcFolder", {});
  descriptions.addWithDefaultLabel(desc);
}

void ElectronEfficiencyPlotter::dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {
  ibooker.setCurrentFolder(theFolder_);

  h_eff_pt_EB_doubleEG_HLT =
      ibooker.book1D("Eff_pt_barrel_DSTdoubleEG", "DSTdoubleEG Eff. vs Pt (barrel)", ptBin_, ptMin_, ptMax_);
  h_eff_pt_EE_doubleEG_HLT =
      ibooker.book1D("Eff_pt_endcap_DSTdoubleEG", "DSTdoubleEG Eff. vs Pt (endcap)", ptBin_, ptMin_, ptMax_);
  h_eff_pt_EB_singlePhoton_HLT =
      ibooker.book1D("Eff_pt_barrel_DSTsinglePhoton", "DSTsinglePhoton Eff. vs Pt (barrel)", ptBin_, ptMin_, ptMax_);
  h_eff_pt_EE_singlePhoton_HLT =
      ibooker.book1D("Eff_pt_endcap_DSTsinglePhoton", "DSTsinglePhoton Eff. vs Pt (endcap)", ptBin_, ptMin_, ptMax_);

  // Axis title
  h_eff_pt_EB_singlePhoton_HLT->setAxisTitle("p_{T} (GeV)", 1);
  h_eff_pt_EE_singlePhoton_HLT->setAxisTitle("p_{T} (GeV)", 1);
  h_eff_pt_EB_doubleEG_HLT->setAxisTitle("p_{T} (GeV)", 1);
  h_eff_pt_EE_doubleEG_HLT->setAxisTitle("p_{T} (GeV)", 1);

  MonitorElement *Numerator_pt_barrel_doubleEG_hlt =
      igetter.get(sourceFolder_ +
                  "/resonanceZ_Tag_pat_Probe_sctElectron_passDoubleEG_DST_"
                  "fireTrigObj_Pt_Barrel");
  MonitorElement *Numerator_pt_endcap_doubleEG_hlt =
      igetter.get(sourceFolder_ +
                  "/resonanceZ_Tag_pat_Probe_sctElectron_passDoubleEG_DST_"
                  "fireTrigObj_Pt_Endcap");
  MonitorElement *Numerator_pt_barrel_singlePhoton_hlt =
      igetter.get(sourceFolder_ +
                  "/resonanceZ_Tag_pat_Probe_sctElectron_passSinglePhoton_DST_"
                  "fireTrigObj_Pt_Barrel");
  MonitorElement *Numerator_pt_endcap_singlePhoton_hlt =
      igetter.get(sourceFolder_ +
                  "/resonanceZ_Tag_pat_Probe_sctElectron_passSinglePhoton_DST_"
                  "fireTrigObj_Pt_Endcap");
  MonitorElement *Denominator_pt_barrel =
      igetter.get(sourceFolder_ + "/resonanceZ_Tag_pat_Probe_sctElectron_Pt_Barrel");
  MonitorElement *Denominator_pt_endcap =
      igetter.get(sourceFolder_ + "/resonanceZ_Tag_pat_Probe_sctElectron_Pt_Endcap");

  if (Numerator_pt_barrel_doubleEG_hlt && Denominator_pt_barrel)
    calculateEfficiency(Numerator_pt_barrel_doubleEG_hlt, Denominator_pt_barrel, h_eff_pt_EB_doubleEG_HLT);
  if (Numerator_pt_endcap_doubleEG_hlt && Denominator_pt_endcap)
    calculateEfficiency(Numerator_pt_endcap_doubleEG_hlt, Denominator_pt_endcap, h_eff_pt_EE_doubleEG_HLT);
  if (Numerator_pt_barrel_singlePhoton_hlt && Denominator_pt_barrel)
    calculateEfficiency(Numerator_pt_barrel_singlePhoton_hlt, Denominator_pt_barrel, h_eff_pt_EB_singlePhoton_HLT);
  if (Numerator_pt_endcap_singlePhoton_hlt && Denominator_pt_endcap)
    calculateEfficiency(Numerator_pt_endcap_singlePhoton_hlt, Denominator_pt_endcap, h_eff_pt_EE_singlePhoton_HLT);
}

void ElectronEfficiencyPlotter::calculateEfficiency(MonitorElement *Numerator,
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

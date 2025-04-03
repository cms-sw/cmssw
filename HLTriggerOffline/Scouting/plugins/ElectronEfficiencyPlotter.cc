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
  const std::vector<std::string> vtriggerSelection_;

  
  std::vector<MonitorElement*> h_eff_sctel_leading_pt_EB;
  std::vector<MonitorElement*> h_eff_sctel_leading_pt_EE;
  std::vector<MonitorElement*> h_eff_patel_leading_pt_EB;
  std::vector<MonitorElement*> h_eff_patel_leading_pt_EE;
  std::vector<MonitorElement*> h_eff_sctel_subleading_pt_EB;
  std::vector<MonitorElement*> h_eff_sctel_subleading_pt_EE;
  std::vector<MonitorElement*> h_eff_patel_subleading_pt_EB;
  std::vector<MonitorElement*> h_eff_patel_subleading_pt_EE;

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
      sourceFolder_{ps.getParameter<string>("srcFolder")},
      vtriggerSelection_{ps.getParameter<std::vector<string>>("triggerSelection")} {}

void ElectronEfficiencyPlotter::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<int>("ptBin", 5);
  desc.add<double>("ptMin", 0);
  desc.add<double>("ptMax", 100);
  desc.add<std::string>("sctElectronID", {});
  desc.add<std::string>("folder", {});
  desc.add<std::string>("srcFolder", {});
  desc.add<std::vector<std::string>>("triggerSelection", {});
  descriptions.addWithDefaultLabel(desc);
}

void ElectronEfficiencyPlotter::dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {
  ibooker.setCurrentFolder(theFolder_);

  int iPicture = 0;
  for (auto const &vt : vtriggerSelection_){
      std::string cleaned_vt = vt;
      cleaned_vt.erase(std::remove(cleaned_vt.begin(), cleaned_vt.end(), '*'), cleaned_vt.end());

      // Leading Electron
      h_eff_sctel_leading_pt_EB.push_back(
          ibooker.book1D("Eff_sctElectron_leading_pt_barrel_" + cleaned_vt, cleaned_vt + " Eff. vs Pt (barrel)", ptBin_, ptMin_, ptMax_)
      );
      h_eff_sctel_leading_pt_EE.push_back(
          ibooker.book1D("Eff_sctElectron_leading_pt_endcap_" + cleaned_vt, cleaned_vt + " Eff. vs Pt (endcap)", ptBin_, ptMin_, ptMax_)
      );
      h_eff_patel_leading_pt_EB.push_back(
          ibooker.book1D("Eff_patElectron_leading_pt_barrel_" + cleaned_vt, cleaned_vt + " Eff. vs Pt (barrel)", ptBin_, ptMin_, ptMax_)
      );
      h_eff_patel_leading_pt_EE.push_back(
          ibooker.book1D("Eff_patElectron_leading_pt_endcap_" + cleaned_vt, cleaned_vt + " Eff. vs Pt (endcap)", ptBin_, ptMin_, ptMax_)
      );
      h_eff_sctel_subleading_pt_EB.push_back(
          ibooker.book1D("Eff_sctElectron_subleading_pt_barrel_" + cleaned_vt, cleaned_vt + " Eff. vs Pt (barrel)", ptBin_, ptMin_, ptMax_)
      );
      h_eff_sctel_subleading_pt_EE.push_back(
          ibooker.book1D("Eff_sctElectron_subleading_pt_endcap_" + cleaned_vt, cleaned_vt + " Eff. vs Pt (endcap)", ptBin_, ptMin_, ptMax_)
      );
      h_eff_patel_subleading_pt_EB.push_back(
          ibooker.book1D("Eff_patElectron_subleading_pt_barrel_" + cleaned_vt, cleaned_vt + " Eff. vs Pt (barrel)", ptBin_, ptMin_, ptMax_)
      );
      h_eff_patel_subleading_pt_EE.push_back(
          ibooker.book1D("Eff_patElectron_subleading_pt_endcap_" + cleaned_vt, cleaned_vt + " Eff. vs Pt (endcap)", ptBin_, ptMin_, ptMax_)
      );


      h_eff_sctel_leading_pt_EB.at(iPicture)->setAxisTitle("p_{T} (GeV)", 1);
      h_eff_sctel_leading_pt_EE.at(iPicture)->setAxisTitle("p_{T} (GeV)", 1);
      h_eff_patel_leading_pt_EB.at(iPicture)->setAxisTitle("p_{T} (GeV)", 1);
      h_eff_patel_leading_pt_EE.at(iPicture)->setAxisTitle("p_{T} (GeV)", 1);
      h_eff_sctel_subleading_pt_EB.at(iPicture)->setAxisTitle("p_{T} (GeV)", 1);
      h_eff_sctel_subleading_pt_EE.at(iPicture)->setAxisTitle("p_{T} (GeV)", 1);
      h_eff_patel_subleading_pt_EB.at(iPicture)->setAxisTitle("p_{T} (GeV)", 1);
      h_eff_patel_subleading_pt_EE.at(iPicture)->setAxisTitle("p_{T} (GeV)", 1);


      
      MonitorElement *Numerator_sctel_leading_pt_barrel = 
         igetter.get(sourceFolder_ + 
                     "/resonanceZ_Tag_pat_Probe_sctElectron_leading_Pt_Barrel_pass" + 
                     cleaned_vt + 
                     "_fireTrigObj"
                    );
      MonitorElement *Denominator_sctel_leading_pt_barrel = 
         igetter.get(sourceFolder_ + 
                     "/resonanceZ_Tag_pat_Probe_sctElectron_leading_Pt_Barrel_passBaseDST" 
                    );
      if (Numerator_sctel_leading_pt_barrel && Denominator_sctel_leading_pt_barrel)
         calculateEfficiency(Numerator_sctel_leading_pt_barrel, Denominator_sctel_leading_pt_barrel, h_eff_sctel_leading_pt_EB.at(iPicture));


      MonitorElement *Numerator_sctel_leading_pt_endcap = 
         igetter.get(sourceFolder_ + 
                     "/resonanceZ_Tag_pat_Probe_sctElectron_leading_Pt_Endcap_pass" + 
                     cleaned_vt + 
                     "_fireTrigObj"
                    );
      MonitorElement *Denominator_sctel_leading_pt_endcap = 
         igetter.get(sourceFolder_ + 
                     "/resonanceZ_Tag_pat_Probe_sctElectron_leading_Pt_Endcap_passBaseDST" 
                    );
      if (Numerator_sctel_leading_pt_endcap && Denominator_sctel_leading_pt_endcap)
         calculateEfficiency(Numerator_sctel_leading_pt_endcap, Denominator_sctel_leading_pt_endcap, h_eff_sctel_leading_pt_EE.at(iPicture));

      MonitorElement *Numerator_sctel_subleading_pt_barrel = 
         igetter.get(sourceFolder_ + 
                     "/resonanceZ_Tag_pat_Probe_sctElectron_subleading_Pt_Barrel_pass" + 
                     cleaned_vt + 
                     "_fireTrigObj"
                    );
      MonitorElement *Denominator_sctel_subleading_pt_barrel = 
         igetter.get(sourceFolder_ + 
                     "/resonanceZ_Tag_pat_Probe_sctElectron_subleading_Pt_Barrel_passBaseDST" 
                    );
      if (Numerator_sctel_subleading_pt_barrel && Denominator_sctel_subleading_pt_barrel)
         calculateEfficiency(Numerator_sctel_subleading_pt_barrel, Denominator_sctel_subleading_pt_barrel, h_eff_sctel_subleading_pt_EB.at(iPicture));


      MonitorElement *Numerator_sctel_subleading_pt_endcap = 
         igetter.get(sourceFolder_ + 
                     "/resonanceZ_Tag_pat_Probe_sctElectron_subleading_Pt_Endcap_pass" + 
                     cleaned_vt + 
                     "_fireTrigObj"
                    );
      MonitorElement *Denominator_sctel_subleading_pt_endcap = 
         igetter.get(sourceFolder_ + 
                     "/resonanceZ_Tag_pat_Probe_sctElectron_subleading_Pt_Endcap_passBaseDST" 
                    );
      if (Numerator_sctel_subleading_pt_endcap && Denominator_sctel_subleading_pt_endcap)
         calculateEfficiency(Numerator_sctel_subleading_pt_endcap, Denominator_sctel_subleading_pt_endcap, h_eff_sctel_subleading_pt_EE.at(iPicture));

      MonitorElement *Numerator_patel_leading_pt_barrel = 
         igetter.get(sourceFolder_ + 
                     "/resonanceZ_Tag_pat_Probe_patElectron_leading_Pt_Barrel_pass" + 
                     cleaned_vt + 
                     "_fireTrigObj"
                    );
      MonitorElement *Denominator_patel_leading_pt_barrel = 
         igetter.get(sourceFolder_ + 
                     "/resonanceZ_Tag_pat_Probe_patElectron_leading_Pt_Barrel_passBaseDST" 
                    );
      if (Numerator_patel_leading_pt_barrel && Denominator_patel_leading_pt_barrel)
         calculateEfficiency(Numerator_patel_leading_pt_barrel, Denominator_patel_leading_pt_barrel, h_eff_patel_leading_pt_EB.at(iPicture));


      MonitorElement *Numerator_patel_leading_pt_endcap = 
         igetter.get(sourceFolder_ + 
                     "/resonanceZ_Tag_pat_Probe_patElectron_leading_Pt_Endcap_pass" + 
                     cleaned_vt + 
                     "_fireTrigObj"
                    );
      MonitorElement *Denominator_patel_leading_pt_endcap = 
         igetter.get(sourceFolder_ + 
                     "/resonanceZ_Tag_pat_Probe_patElectron_leading_Pt_Endcap_passBaseDST" 
                    );
      if (Numerator_patel_leading_pt_endcap && Denominator_patel_leading_pt_endcap)
         calculateEfficiency(Numerator_patel_leading_pt_endcap, Denominator_patel_leading_pt_endcap, h_eff_patel_leading_pt_EE.at(iPicture));

      MonitorElement *Numerator_patel_subleading_pt_barrel = 
         igetter.get(sourceFolder_ + 
                     "/resonanceZ_Tag_pat_Probe_patElectron_subleading_Pt_Barrel_pass" + 
                     cleaned_vt + 
                     "_fireTrigObj"
                    );
      MonitorElement *Denominator_patel_subleading_pt_barrel = 
         igetter.get(sourceFolder_ + 
                     "/resonanceZ_Tag_pat_Probe_patElectron_subleading_Pt_Barrel_passBaseDST" 
                    );
      if (Numerator_patel_subleading_pt_barrel && Denominator_patel_subleading_pt_barrel)
         calculateEfficiency(Numerator_patel_subleading_pt_barrel, Denominator_patel_subleading_pt_barrel, h_eff_patel_subleading_pt_EB.at(iPicture));


      MonitorElement *Numerator_patel_subleading_pt_endcap = 
         igetter.get(sourceFolder_ + 
                     "/resonanceZ_Tag_pat_Probe_patElectron_subleading_Pt_Endcap_pass" + 
                     cleaned_vt + 
                     "_fireTrigObj"
                    );
      MonitorElement *Denominator_patel_subleading_pt_endcap = 
         igetter.get(sourceFolder_ + 
                     "/resonanceZ_Tag_pat_Probe_patElectron_subleading_Pt_Endcap_passBaseDST" 
                    );
      if (Numerator_patel_subleading_pt_endcap && Denominator_patel_subleading_pt_endcap)
         calculateEfficiency(Numerator_patel_subleading_pt_endcap, Denominator_patel_subleading_pt_endcap, h_eff_patel_subleading_pt_EE.at(iPicture));


      iPicture += 1;
  }
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

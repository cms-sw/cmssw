#ifndef SUSY_HLT_Razor_H
#define SUSY_HLT_Razor_H

//event
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

//DQM
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

// MET
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

// Jets
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

// Trigger
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"

//Hemispheres
#include "HLTrigger/JetMET/interface/HLTRHemisphere.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "TLorentzVector.h"

class SUSY_HLT_Razor: public DQMEDAnalyzer{

  public:
  SUSY_HLT_Razor(const edm::ParameterSet& ps);
  static double CalcMR(TLorentzVector ja,TLorentzVector jb);
  static double CalcR(double MR, TLorentzVector ja,TLorentzVector jb, edm::Handle<edm::View<reco::MET> > met, const std::vector<math::XYZTLorentzVector>& muons);
  virtual ~SUSY_HLT_Razor();
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  protected:
  void dqmBeginRun(edm::Run const &, edm::EventSetup const &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const& e, edm::EventSetup const& eSetup);
  void beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& eSetup) ;
  void endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& eSetup);
  void endRun(edm::Run const& run, edm::EventSetup const& eSetup);

  private:
  //histos booking function
  void bookHistos(DQMStore::IBooker &);
  
  //variables from config file
  edm::EDGetTokenT<edm::View<reco::MET> > theMETCollection_;
  edm::EDGetTokenT<edm::TriggerResults> triggerResults_;
  edm::EDGetTokenT<trigger::TriggerEvent> theTrigSummary_;
  edm::EDGetTokenT<edm::View<reco::Jet> > theJetCollection_;
  edm::EDGetTokenT<std::vector<math::XYZTLorentzVector> > theHemispheres_;

  std::string triggerPath_;
  std::string denomPath_; // denominator path for 1.4e34 lumi
  std::string denomPathLoose_; // denominator path for 7e33 lumi
  edm::InputTag triggerFilter_;
  edm::InputTag caloFilter_;
  
  // Histograms
  MonitorElement* h_mr;
  MonitorElement* h_rsq;
  MonitorElement* h_mrRsq;
  MonitorElement* h_mr_denom;
  MonitorElement* h_rsq_denom;
  MonitorElement* h_mrRsq_denom;
  MonitorElement* h_mr_tight;
  MonitorElement* h_rsq_tight;
  MonitorElement* h_mr_tight_denom;
  MonitorElement* h_rsq_tight_denom;
  MonitorElement* h_rsq_loose;  
  MonitorElement* h_rsq_loose_denom;  
  MonitorElement* h_ht;
  MonitorElement* h_met;
  MonitorElement* h_htMet;
  MonitorElement* h_ht_denom;
  MonitorElement* h_met_denom;
  MonitorElement* h_htMet_denom;
  MonitorElement* h_online_mr_vs_mr;
  MonitorElement* h_online_rsq_vs_rsq;
  MonitorElement* h_online_mr_vs_mr_all;
  MonitorElement* h_online_rsq_vs_rsq_all;
  MonitorElement* h_calo_mr_vs_mr;
  MonitorElement* h_calo_rsq_vs_rsq;
  MonitorElement* h_calo_mr_vs_mr_all;
  MonitorElement* h_calo_rsq_vs_rsq_all;

};

#endif

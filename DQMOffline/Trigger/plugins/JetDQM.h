#ifndef DQMOffline_Trigger_JetDQM_h
#define DQMOffline_Trigger_JetDQM_h

#include <vector>

#include "DQMOffline/Trigger/plugins/TriggerDQMBase.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/METReco/interface/PFMET.h"

class JetDQM : public TriggerDQMBase {
public:
  JetDQM();
  ~JetDQM() override;

  void initialise(const edm::ParameterSet& iConfig);
  void bookHistograms(DQMStore::IBooker&);
  void fillHistograms(const std::vector<reco::PFJet>& jets, const reco::PFMET& pfmet, const int ls, const bool passCond);
  static void fillJetDescription(edm::ParameterSetDescription& histoPSet);

private:
  std::vector<double> jetpt_variable_binning_;
  std::vector<double> jet1pt_variable_binning_;
  std::vector<double> jet2pt_variable_binning_;
  std::vector<double> mjj_variable_binning_;

  MEbinning jeteta_binning_;
  MEbinning detajj_binning_;
  MEbinning dphijj_binning_;
  MEbinning mindphijmet_binning_;
  MEbinning ls_binning_;

  // leading jets pT and eta
  ObjME jet1ptME_;
  ObjME jet2ptME_;
  ObjME jet1etaME_;
  ObjME jet2etaME_;

  // most central and most forward jets pT and eta
  ObjME cjetetaME_;
  ObjME fjetetaME_;
  ObjME cjetptME_;
  ObjME fjetptME_;

  // leading pair quantities
  ObjME mjjME_;
  ObjME detajjME_;
  ObjME dphijjME_;

  // correlations MET-jets
  ObjME mindphijmetME_;

  ObjME jet1etaVsLS_;
  ObjME mjjVsLS_;
  ObjME mindphijmetVsLS_;
};

#endif

#ifndef DQMOffline_Trigger_JetDQM_H
#define DQMOffline_Trigger_JetDQM_H

#include <vector>

#include "DQMOffline/Trigger/plugins/GENERICDQM.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class JetDQM : public GENERICDQM
{
 public:
  JetDQM();
  virtual ~JetDQM();

  void initialise(const edm::ParameterSet& iConfig);
  void bookHistograms(DQMStore::IBooker &);
  void fillHistograms(const std::vector<reco::PFJet> & jets,
		      const reco::PFMET & pfmet,
		      const int & ls,
		      const bool passCond);
  static void fillJetDescription(edm::ParameterSetDescription & histoPSet);

  //keep public to fill them directly....
  std::vector<double> jetpt_variable_binning_;
  std::vector<double> jet1pt_variable_binning_;
  std::vector<double> jet2pt_variable_binning_;
  std::vector<double> mjj_variable_binning_;
  MEbinning           jeteta_binning_;
  MEbinning           detajj_binning_;
  MEbinning           dphijj_binning_;
  MEbinning           mindphijmet_binning_;

  MEbinning           ls_binning_;

private:
  //leading jets pT and eta
  OBJME jet1ptME_;
  OBJME jet2ptME_;
  OBJME jet1etaME_;
  OBJME jet2etaME_;
  //most central and most forward jets pT and eta
  OBJME cjetetaME_;
  OBJME fjetetaME_;
  OBJME cjetptME_;
  OBJME fjetptME_;
  //leading pair quantities
  OBJME mjjME_;
  OBJME detajjME_;
  OBJME dphijjME_;
  //correlations MET-jets
  OBJME mindphijmetME_;

  OBJME jet1etaVsLS_;
  OBJME mjjVsLS_;
  OBJME mindphijmetVsLS_;

};//class

#endif //DQMOffline_Trigger_JetDQM_H

#ifndef DQMOffline_Trigger_HTDQM_h
#define DQMOffline_Trigger_HTDQM_h

#include "DQMOffline/Trigger/plugins/TriggerDQMBase.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/JetReco/interface/PFJet.h"

class HTDQM : public TriggerDQMBase {
public:
  HTDQM();
  ~HTDQM() override;

  void initialise(const edm::ParameterSet& iConfig);
  void bookHistograms(DQMStore::IBooker&);
  void fillHistograms(const std::vector<reco::PFJet>& htjets, const double& met, const int& ls, const bool passCond);
  static void fillHtDescription(edm::ParameterSetDescription& histoPSet);

private:
  std::vector<double> ht_variable_binning_;
  std::vector<double> met_variable_binning_;
  MEbinning ht_binning_;
  MEbinning ls_binning_;

  ObjME htME_variableBinning_;
  ObjME htVsMET_;
  ObjME htVsLS_;
};

#endif

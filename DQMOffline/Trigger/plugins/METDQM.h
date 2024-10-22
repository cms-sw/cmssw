#ifndef DQMOffline_Trigger_METDQM_H
#define DQMOffline_Trigger_METDQM_H

#include "DQMOffline/Trigger/plugins/TriggerDQMBase.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class METDQM : public TriggerDQMBase {
public:
  METDQM();
  ~METDQM() override;

  void initialise(const edm::ParameterSet& iConfig);
  void bookHistograms(DQMStore::IBooker&);
  void fillHistograms(const double& met, const double& phi, const int& ls, const bool passCond);
  static void fillMetDescription(edm::ParameterSetDescription& histoPSet);

private:
  std::vector<double> met_variable_binning_;
  MEbinning met_binning_;
  MEbinning phi_binning_;
  MEbinning ls_binning_;

  ObjME metME_;
  ObjME metME_variableBinning_;
  ObjME metVsLS_;
  ObjME metPhiME_;
};

#endif

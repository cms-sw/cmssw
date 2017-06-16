#ifndef DQMOffline_Trigger_METDQM_H
#define DQMOffline_Trigger_METDQM_H

#include "DQMOffline/Trigger/plugins/GENERICDQM.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class METDQM : public GENERICDQM
{
 public:
  METDQM();
  virtual ~METDQM();

  void initialise(const edm::ParameterSet& iConfig);
  void bookHistograms(DQMStore::IBooker &);
  void fillHistograms(const double & met,
		      const double & phi,
		      const int & ls,
		      const bool passCond);
  static void fillMetDescription(edm::ParameterSetDescription & histoPSet);

  //keep public to fill them directly....
  std::vector<double> met_variable_binning_;
  MEbinning           met_binning_;
  MEbinning           phi_binning_;
  MEbinning           ls_binning_;

private:
  OBJME metME_;
  OBJME metME_variableBinning_;
  OBJME metVsLS_;
  OBJME metPhiME_;

};//class

#endif //DQMOffline_Trigger_METDQM_H

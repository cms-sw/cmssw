#ifndef DQMOffline_Trigger_HMesonGammaDQM_h
#define DQMOffline_Trigger_HMesonGammaDQM_h

#include <vector>
#include "TLorentzVector.h"

#include "DQMOffline/Trigger/plugins/TriggerDQMBase.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

class HMesonGammaDQM : public TriggerDQMBase {
public:
  HMesonGammaDQM();
  ~HMesonGammaDQM() override;

  void initialise(const edm::ParameterSet& iConfig);
  void bookHistograms(DQMStore::IBooker&);
  void fillHistograms(const reco::PhotonCollection& photons,
                      const std::vector<TLorentzVector>& mesons,
                      const int ls,
                      const bool passCond);
  static void fillHmgDescription(edm::ParameterSetDescription& histoPSet);

private:
  std::vector<double> gammapt_variable_binning_;
  std::vector<double> mesonpt_variable_binning_;

  MEbinning eta_binning_;
  MEbinning ls_binning_;

  //leading gamma/meson pT and eta
  ObjME gammaptME_;
  ObjME mesonptME_;
  ObjME gammaetaME_;
  ObjME mesonetaME_;
  ObjME gammaetaVsLS_;
};

#endif

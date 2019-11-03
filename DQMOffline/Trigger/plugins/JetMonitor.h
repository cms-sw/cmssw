#ifndef JETMETMONITOR_H
#define JETMETMONITOR_H

#include <string>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMOffline/Trigger/interface/TriggerDQMBase.h"
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"

class JetMonitor : public DQMEDAnalyzer, public TriggerDQMBase {

 public:
  typedef dqm::reco::MonitorElement MonitorElement;
  typedef dqm::reco::DQMStore DQMStore;

  JetMonitor(const edm::ParameterSet&);
  ~JetMonitor() throw() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;

  bool isBarrel(double eta);
  bool isEndCapP(double eta);
  bool isEndCapM(double eta);
  bool isForward(double eta);
  bool isHEP17(double eta, double phi);
  bool isHEM17(double eta, double phi);
  bool isHEP18(double eta, double phi);  // -0.87< Phi < -1.22

  void bookMESub(DQMStore::IBooker&, ObjME* a_me, const int len_, const std::string& h_Name, const std::string& h_Title, const std::string& h_subOptName, const std::string& h_subOptTitle, const bool doPhi=true, const bool doEta=true, const bool doEtaPhi=true, const bool doVsLS=true);
  void FillME(ObjME* a_me, const double pt_, const double phi_, const double eta_, const int ls_, const std::string& denu, const bool doPhi=true, const bool doEta=true, const bool doEtaPhi=true, const bool doVsLS=true);

 private:
  const std::string folderName_;

  const bool requireValidHLTPaths_;
  bool hltPathsAreValid_;

  double ptcut_;
  bool isPFJetTrig;
  bool isCaloJetTrig;

  edm::EDGetTokenT<edm::View<reco::Jet> > jetSrc_;

  std::unique_ptr<GenericTriggerEventFlag> num_genTriggerEventFlag_;
  std::unique_ptr<GenericTriggerEventFlag> den_genTriggerEventFlag_;

  MEbinning jetpt_binning_;
  MEbinning jetptThr_binning_;
  MEbinning ls_binning_;

  ObjME a_ME[7];
  ObjME a_ME_HB[7];
  ObjME a_ME_HE[7];
  ObjME a_ME_HF[7];
  ObjME a_ME_HE_p[7];
  ObjME a_ME_HE_m[7];
  ObjME a_ME_HEM17[7];
  ObjME a_ME_HEP17[7];
  ObjME a_ME_HEP18[7];

  ObjME jetHEP17_AbsEtaVsPhi_;
  ObjME jetHEM17_AbsEtaVsPhi_;
  ObjME jetHEP17_AbsEta_;
  ObjME jetHEM17_AbsEta_;

  std::vector<double> v_jetpt;
  std::vector<double> v_jeteta;
  std::vector<double> v_jetphi;

  // (mia) not optimal, we should make use of variable binning which reflects the detector !
  MEbinning jet_phi_binning_{32, -3.2, 3.2};
  MEbinning jet_eta_binning_{20, -5, 5};

  MEbinning eta_binning_hep17_{9,  1.3,  3.0};
  MEbinning eta_binning_hem17_{9, -3.0, -1.3};

  MEbinning phi_binning_hep17_{7, -0.87, -0.52};
  MEbinning phi_binning_hep18_{7, -0.52, -0.17};
};

#endif

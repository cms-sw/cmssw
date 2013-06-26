// system include files
#include <memory>

// Root objects
#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TProfile.h"
#include "TDirectory.h"
#include "TTree.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "Calibration/IsolatedParticles/interface/TrackSelection.h"

class IsoTrig : public edm::EDAnalyzer {

public:
  explicit IsoTrig(const edm::ParameterSet&);
  ~IsoTrig();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  double dR(double eta1, double eta2, double phi1, double phi2);
  //  std::vector<double> var[3][3];

private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  virtual void beginRun(edm::Run const&, edm::EventSetup const&);
  virtual void endRun(edm::Run const&, edm::EventSetup const&);
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
  void fillHist(int, math::XYZTLorentzVector&);
  void fillDifferences(int, math::XYZTLorentzVector&, math::XYZTLorentzVector&, bool);
  void fillCuts(int, double, double, double, math::XYZTLorentzVector&, int, bool);
  double dPt(math::XYZTLorentzVector&, math::XYZTLorentzVector&);
  double dP(math::XYZTLorentzVector&, math::XYZTLorentzVector&);
  double dinvPt(math::XYZTLorentzVector&, math::XYZTLorentzVector&);
  double dEta(math::XYZTLorentzVector&, math::XYZTLorentzVector&);
  double dPhi(math::XYZTLorentzVector&, math::XYZTLorentzVector&);
  double dR(math::XYZTLorentzVector&, math::XYZTLorentzVector&);

  // ----------member data ---------------------------
  HLTConfigProvider          hltConfig_;
  std::string                Det;
  int                        verbosity;
  spr::trackSelectionParameters selectionParameters;
  std::string                theTrackQuality;
  double                     dr_L1, a_mipR, a_coneR, a_charIsoR, a_neutIsoR;
  double                     a_neutR1, a_neutR2, cutMip, cutCharge, cutNeutral;
  int                        minRunNo, maxRunNo;
  std::map<unsigned int, unsigned int> TrigList;
  std::map<unsigned int, const std::pair<int, int>> TrigPreList;
  bool                       changed;
  edm::Service<TFileService> fs;
  TH1I *h_nHLT, *h_HLT, *h_PreL1, *h_PreHLT, *h_Pre, *h_nL3Objs;
  TH1D *h_PreL1wt, *h_PreHLTwt, *h_L1ObjEnergy;
  TH1D *h_p[20], *h_pt[20], *h_eta[20], *h_phi[20];
  TH1D *h_dEtaL1[2], *h_dPhiL1[2], *h_dRL1[2], *h_etaMipTracks[2][2];
  TH1D *h_dEta[6], *h_dPhi[6], *h_dPt[6], *h_dP[6], *h_dinvPt[6], *h_mindR[6];
  TH1D *h_eMip[2], *h_eMaxNearP[2], *h_eNeutIso[2], *h_etaCalibTracks[2][2];
  TH1I *g_Pre, *g_PreL1, *g_PreHLT, *g_Accepts;
};

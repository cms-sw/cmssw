
// system include files
#include <memory>

// Root objects
#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TProfile.h"
#include "TDirectory.h"
#include "TH1.h"
#include "TH2.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "Calibration/IsolatedParticles/interface/TrackSelection.h"

class StudyHLT : public edm::EDAnalyzer {

public:
  explicit StudyHLT(const edm::ParameterSet&);
  ~StudyHLT();

private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  virtual void beginRun(edm::Run const&, edm::EventSetup const&);
  virtual void endRun(edm::Run const&, edm::EventSetup const&);
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
  void fillTrack(int, double, double, double, double);
  void fillIsolation(int, double, double, double);

  // ----------member data ---------------------------
  HLTConfigProvider          hltConfig_;
  edm::Service<TFileService> fs;
  int                        verbosity;
  spr::trackSelectionParameters selectionParameters;
  std::vector<std::string>   trigNames, HLTNames;
  std::string                theTrackQuality;
  double                     minTrackP, maxTrackEta, tMinE_, tMaxE_, tMinH_, tMaxH_;
  bool                       changed, firstEvent;
  TH1I                      *h_nHLT, *h_HLTAccept;
  TH2I                      *h_nHLTvsRN;
  std::vector<TH1I*>         h_HLTAccepts;
  TH1D                      *h_p[8], *h_pt[8], *h_eta[8], *h_phi[8];
  TH1I                      *h_ntrk[2];
  TH1D                      *h_maxNearP[2], *h_ene1[2], *h_ene2[2], *h_ediff[2];
  int                        nRun;
};

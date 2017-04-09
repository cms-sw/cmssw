
// system include files
#include <memory>
#include <string>

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

#include "DataFormats/Luminosity/interface/LumiDetails.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

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
  void fillEnergy(int, int, double, double, double, double, double);
  std::string truncate_str(const std::string&);

  // ----------member data ---------------------------
  static const int           nPBin=10, nEtaBin=4, nPVBin=4;
  HLTConfigProvider          hltConfig_;
  edm::Service<TFileService> fs;
  int                        verbosity;
  spr::trackSelectionParameters selectionParameters;
  std::vector<std::string>   trigNames, HLTNames;
  std::string                theTrackQuality;
  double                     minTrackP, maxTrackEta, tMinE_, tMaxE_, tMinH_, tMaxH_;
  bool                       isItAOD, changed, firstEvent;

  edm::InputTag              triggerEvent_, theTriggerResultsLabel;
  edm::EDGetTokenT<LumiDetails>                       tok_lumi;
  edm::EDGetTokenT<trigger::TriggerEvent>             tok_trigEvt;
  edm::EDGetTokenT<edm::TriggerResults>               tok_trigRes;
 
  edm::EDGetTokenT<reco::TrackCollection>             tok_genTrack_;
  edm::EDGetTokenT<reco::VertexCollection>            tok_recVtx_;
  edm::EDGetTokenT<EcalRecHitCollection>              tok_EB_;
  edm::EDGetTokenT<EcalRecHitCollection>              tok_EE_;
  edm::EDGetTokenT<HBHERecHitCollection>              tok_hbhe_;

  TH1I                      *h_nHLT, *h_HLTAccept, *h_HLTCorr, *h_numberPV;
  TH1I                      *h_goodPV, *h_goodRun;
  TH2I                      *h_nHLTvsRN;
  std::vector<TH1I*>         h_HLTAccepts;
  TH1D                      *h_p[nPVBin+8], *h_pt[nPVBin+8], *h_eta[nPVBin+8], *h_phi[nPVBin+8];
  TH1I                      *h_ntrk[2];
  TH1D                      *h_maxNearP[2], *h_ene1[2], *h_ene2[2], *h_ediff[2];
  TH1D                      *h_energy[nPVBin+4][nPBin][nEtaBin][6];
  int                        nRun, etaBin[nEtaBin+1], pvBin[nPVBin+1];
  double                     pBin[nPBin+1];
};

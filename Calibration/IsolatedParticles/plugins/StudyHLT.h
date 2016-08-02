
// system include files
#include <memory>
#include <string>

// Root objects
#include "TH1.h"
#include "TH2.h"
#include "TTree.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

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

class StudyHLT : public edm::one::EDAnalyzer<edm::one::WatchRuns,edm::one::SharedResources> {

public:
  explicit StudyHLT(const edm::ParameterSet&);
  ~StudyHLT();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void analyze(edm::Event const&, edm::EventSetup const&) override;
  virtual void beginJob() ;
  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

  void clear();
  void fillTrack(int, double, double, double, double);
  void fillIsolation(int, double, double, double);
  void fillEnergy(int, int, double, double, double, double, double);
  std::string truncate_str(const std::string&);

  // ----------member data ---------------------------
  static const int           nPBin=10, nEtaBin=4, nPVBin=4;
  HLTConfigProvider          hltConfig_;
  edm::Service<TFileService> fs_;
  int                        verbosity_;
  spr::trackSelectionParameters selectionParameters_;
  std::vector<std::string>   trigNames_, HLTNames_;
  std::string                theTrackQuality_;
  double                     minTrackP_, maxTrackEta_;
  double                     tMinE_, tMaxE_, tMinH_, tMaxH_;
  bool                       isItAOD_, changed_, firstEvent_, doTree_;

  edm::InputTag              triggerEvent_, theTriggerResultsLabel_;
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
  TH1D                      *h_p[nPVBin+8], *h_pt[nPVBin+8], *h_eta[nPVBin+8];
  TH1D                      *h_phi[nPVBin+8];
  TH1I                      *h_ntrk[2];
  TH1D                      *h_maxNearP[2], *h_ene1[2], *h_ene2[2], *h_ediff[2];
  TH1D                      *h_energy[nPVBin+4][nPBin][nEtaBin][6];
  TTree                     *tree_;
  int                        nRun, etaBin[nEtaBin+1], pvBin[nPVBin+1];
  double                     pBin[nPBin+1];
  int                        tr_goodPV, tr_goodRun;
  std::vector<std::string>   tr_TrigName;
  std::vector<double>        tr_TrkPt, tr_TrkP, tr_TrkEta, tr_TrkPhi;
  std::vector<double>        tr_MaxNearP31X31, tr_MaxNearHcalP7x7;
  std::vector<double>        tr_H3x3, tr_H5x5, tr_H7x7;
  std::vector<double>        tr_FE7x7P, tr_FE11x11P, tr_FE15x15P;
  std::vector<bool>          tr_SE7x7P, tr_SE11x11P, tr_SE15x15P;
  std::vector<int>           tr_iEta;
};

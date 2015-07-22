#ifndef EwkMuDQM_H
#define EwkMuDQM_H

/** \class EwkMuDQM
 *
 *  DQM offline for EWKMu
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "HLTrigger/HLTcore/interface/HLTPrescaleProvider.h"

// #include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

namespace reco {
class Muon;
class Jet;
class MET;
class Vertex;
class Photon;
class BeamSpot;
}

class DQMStore;
class MonitorElement;

class EwkMuDQM : public DQMEDAnalyzer {
 public:
  EwkMuDQM(const edm::ParameterSet&);
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  //Book histograms
  void bookHistograms(DQMStore::IBooker &,
    edm::Run const &, edm::EventSetup const &) override;
  virtual void dqmBeginRun(const edm::Run&, const edm::EventSetup&);
  virtual void endRun(const edm::Run&, const edm::EventSetup&);

  void init_histograms();

 private:
  edm::InputTag metTag_;
  edm::InputTag jetTag_;
  edm::EDGetTokenT<edm::TriggerResults> trigTag_;
  edm::EDGetTokenT<edm::View<reco::Muon> > muonTag_;
  edm::EDGetTokenT<edm::View<reco::MET> > metToken_;
  edm::EDGetTokenT<edm::View<reco::Jet> > jetToken_;
  edm::EDGetTokenT<edm::View<reco::Photon> > phoTag_;
  edm::EDGetTokenT<edm::View<reco::Vertex> > vertexTag_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotTag_;
  std::vector<std::string> trigPathNames_;

  bool isAlsoTrackerMuon_;
  double dxyCut_;
  double normalizedChi2Cut_;
  int trackerHitsCut_;
  int pixelHitsCut_;
  int muonHitsCut_;
  int nMatchesCut_;

  bool isRelativeIso_;
  bool isCombinedIso_;
  double isoCut03_;

  double acopCut_;
  double metMin_;
  double metMax_;
  double mtMin_;
  double mtMax_;

  double ptCut_;
  double etaCut_;

  double ptThrForZ1_;
  double ptThrForZ2_;

  double dimuonMassMin_;
  double dimuonMassMax_;

  double eJetMin_;
  int nJetMax_;

  double ptThrForPhoton_;
  int nPhoMax_;

  bool isValidHltConfig_;
  HLTPrescaleProvider hltPrescaleProvider_;

  unsigned int nall;
  unsigned int nrec;
  unsigned int niso;
  unsigned int nhlt;
  unsigned int nmet;
  unsigned int nsel;
  unsigned int nz;

  MonitorElement* pt_before_;
  MonitorElement* pt_after_;
  MonitorElement* eta_before_;
  MonitorElement* eta_after_;
  MonitorElement* dxy_before_;
  MonitorElement* dxy_after_;
  MonitorElement* goodewkmuon_before_;
  MonitorElement* goodewkmuon_after_;
  MonitorElement* iso_before_;
  MonitorElement* iso_after_;
  MonitorElement* trig_before_;
  MonitorElement* trig_after_;
  MonitorElement* mt_before_;
  MonitorElement* mt_after_;
  MonitorElement* met_before_;
  MonitorElement* met_after_;
  MonitorElement* acop_before_;
  MonitorElement* acop_after_;

  MonitorElement* njets_before_;
  MonitorElement* njets_after_;
  MonitorElement* njets_afterZ_;
  MonitorElement* leadingjet_pt_before_;
  MonitorElement* leadingjet_pt_after_;
  MonitorElement* leadingjet_pt_afterZ_;
  MonitorElement* leadingjet_eta_before_;
  MonitorElement* leadingjet_eta_after_;
  MonitorElement* leadingjet_eta_afterZ_;

  // MonitorElement* ptPlus_before_;
  // MonitorElement* ptMinus_before_;
  MonitorElement* ptDiffPM_before_;
  // MonitorElement* ptPlus_afterW_;
  // MonitorElement* ptMinus_afterW_;
  // MonitorElement* ptPlus_afterZ_;
  // MonitorElement* ptMinus_afterZ_;
  MonitorElement* ptDiffPM_afterZ_;

  MonitorElement* met_afterZ_;
  MonitorElement* pt1_afterZ_;
  MonitorElement* eta1_afterZ_;
  MonitorElement* dxy1_afterZ_;
  MonitorElement* goodewkmuon1_afterZ_;
  MonitorElement* iso1_afterZ_;
  MonitorElement* pt2_afterZ_;
  MonitorElement* eta2_afterZ_;
  MonitorElement* dxy2_afterZ_;
  MonitorElement* goodewkmuon2_afterZ_;
  MonitorElement* iso2_afterZ_;

  // filled if there is a Z-candidate
  MonitorElement* n_zselPt1thr_;  // number of muons in the event with pt>pt1thr
  MonitorElement* n_zselPt2thr_;  // number of muons in the event with pt>pt2thr

  MonitorElement* ztrig_afterZ_;
  MonitorElement* dimuonmass_before_;
  MonitorElement* dimuonmass_afterZ_;

  MonitorElement* npvs_before_;
  MonitorElement* npvs_after_;
  MonitorElement* npvs_afterZ_;

  MonitorElement* muoncharge_before_;
  MonitorElement* muoncharge_after_;
  MonitorElement* muoncharge_afterZ_;

  MonitorElement* nmuons_;
  MonitorElement* ngoodmuons_;

  MonitorElement* npfph_;
  MonitorElement* nph_;
  MonitorElement* pfphPt_;
  MonitorElement* phPt_;
  MonitorElement* pfphEta_;
  MonitorElement* phEta_;
};

#endif

// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:

#ifndef EwkMuDQM_H
#define EwkMuDQM_H

/** \class EwkMuDQM
 *
 *  DQM offline for EWKMu
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"

class DQMStore;
class MonitorElement;
class EwkMuDQM : public edm::EDAnalyzer {
public:
  EwkMuDQM (const edm::ParameterSet &);
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginJob();
  virtual void endJob();
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
  virtual void endRun(const edm::Run&, const edm::EventSetup&);

  void init_histograms();
private:

  edm::InputTag trigTag_;
  edm::InputTag muonTag_;
  edm::InputTag metTag_;
  //bool metIncludesMuons_; plain met not supported anymore, default is pfMet
  edm::InputTag jetTag_;
  edm::InputTag vertexTag_;

  double ptCut_;
  double etaCut_;
  bool isRelativeIso_;
  bool isCombinedIso_;
  double isoCut03_;
  double mtMin_;
  double mtMax_;
  double metMin_;
  double metMax_;
  double acopCut_;

  double dxyCut_;
  double normalizedChi2Cut_;
  int trackerHitsCut_;
  int pixelHitsCut_;
  int muonHitsCut_;
  bool isAlsoTrackerMuon_;
  int nMatchesCut_;  

  double ptThrForZ1_;
  double ptThrForZ2_;

  double eJetMin_;
  int nJetMax_;

  bool isValidHltConfig_;
  HLTConfigProvider  hltConfigProvider_;


  unsigned int nall;
  unsigned int nrec;
  unsigned int niso;
  unsigned int nhlt;
  unsigned int nmet;
  unsigned int nsel;


  DQMStore* theDbe;

  MonitorElement* pt_before_;
  MonitorElement* pt_after_;

  MonitorElement* eta_before_;
  MonitorElement* eta_after_;

  MonitorElement* dxy_before_;
  MonitorElement* dxy_after_;

  MonitorElement* chi2_before_;
  MonitorElement* chi2_after_;

  MonitorElement* nhits_before_;
  MonitorElement* nhits_after_;

  MonitorElement* muonhits_before_;
  MonitorElement* muonhits_after_;

  MonitorElement* goodewkmuon_before_;
  MonitorElement* goodewkmuon_after_;

  MonitorElement* tkmu_before_;
  MonitorElement* tkmu_after_;

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

  MonitorElement* nz1_before_;
  MonitorElement* nz1_after_;

  MonitorElement* nz2_before_;
  MonitorElement* nz2_after_;

  MonitorElement* njets_before_;
  MonitorElement* njets_after_;

  MonitorElement* dimuonmass_before_;
  MonitorElement* dimuonmass_after_;

  MonitorElement* dimuonSAmass_before_;
  MonitorElement* dimuonSAmass_after_;

  MonitorElement* dimuonSASAmass_before_;
  MonitorElement* dimuonSASAmass_after_;

  MonitorElement* npvs_before_;
  MonitorElement* npvs_after_;

  MonitorElement* muoncharge_before_;
  MonitorElement* muoncharge_after_;

  MonitorElement* ptmuonZ_after_;
};

#endif

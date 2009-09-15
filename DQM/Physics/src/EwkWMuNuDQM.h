#ifndef EwkWMuNuDQM_H
#define EwkWMuNuDQM_H

/** \class EwkWMuNuDQM
 *
 *  DQM offline for EWK WMuNu
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

class DQMStore;
class MonitorElement;
class EwkWMuNuDQM : public edm::EDAnalyzer {
public:
  EwkWMuNuDQM (const edm::ParameterSet &);
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginJob(const edm::EventSetup&);
  virtual void endJob();
  void init_histograms();
private:

  edm::InputTag trigTag_;
  edm::InputTag muonTag_;
  edm::InputTag metTag_;
  bool metIncludesMuons_;
  edm::InputTag jetTag_;

  const std::string muonTrig_;
  bool useTrackerPt_;
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
  bool isAlsoTrackerMuon_;

  double ptThrForZ1_;
  double ptThrForZ2_;

  double eJetMin_;
  int nJetMax_;

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
};

#endif

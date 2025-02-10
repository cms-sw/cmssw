/*
Class definition for ScoutingMuonTagProbeAnalyzer.cc. Declares each
histogram (MonitorElement), numerator and denominator histogram structure
(kProbeKinematicMuonHistos), and any functions used in
ScoutingMuonTagProbeAnalyzer.cc. Also declares the token to read the
scouting muon and scouting vertex collections.

Author: Javier Garcia de Castro, email:javigdc@bu.edu
*/

// Files to include
#ifndef DQMOffline_Scouting_ScoutingMuonTagProbeAnalyzer_h
#define DQMOffline_Scouting_ScoutingMuonTagProbeAnalyzer_h
#include <iostream>
#include <string>
#include <vector>

#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/Scouting/interface/Run3ScoutingMuon.h"
#include "DataFormats/Scouting/interface/Run3ScoutingVertex.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

struct kProbeKinematicMuonHistos {
  dqm::reco::MonitorElement* hPt;
  dqm::reco::MonitorElement* hEta;
  dqm::reco::MonitorElement* hPhi;
  dqm::reco::MonitorElement* hInvMass;
  dqm::reco::MonitorElement* hNormChisq;
  dqm::reco::MonitorElement* hTrk_dxy;
  dqm::reco::MonitorElement* hTrk_dz;
  dqm::reco::MonitorElement* hnPixel;
  dqm::reco::MonitorElement* hnTracker;
  dqm::reco::MonitorElement* htrk_qoverp;
  dqm::reco::MonitorElement* htype;
  dqm::reco::MonitorElement* hcharge;
  dqm::reco::MonitorElement* hecalIso;
  dqm::reco::MonitorElement* hhcalIso;
  dqm::reco::MonitorElement* htrackIso;
  dqm::reco::MonitorElement* hnValidStandAloneMuonHits;
  dqm::reco::MonitorElement* hnStandAloneMuonMatchedStations;
  dqm::reco::MonitorElement* hnValidRecoMuonHits;
  dqm::reco::MonitorElement* hnRecoMuonChambers;
  dqm::reco::MonitorElement* hnRecoMuonChambersCSCorDT;
  dqm::reco::MonitorElement* hnRecoMuonMatches;
  dqm::reco::MonitorElement* hnRecoMuonMatchedStations;
  dqm::reco::MonitorElement* hnRecoMuonExpectedMatchedStations;
  dqm::reco::MonitorElement* hnValidPixelHits;
  dqm::reco::MonitorElement* hnValidStripHits;
  dqm::reco::MonitorElement* hnPixelLayersWithMeasurement;
  dqm::reco::MonitorElement* hnTrackerLayersWithMeasurement;
  dqm::reco::MonitorElement* htrk_chi2;
  dqm::reco::MonitorElement* htrk_ndof;
  dqm::reco::MonitorElement* htrk_lambda;
  dqm::reco::MonitorElement* htrk_pt;
  dqm::reco::MonitorElement* htrk_eta;
  dqm::reco::MonitorElement* htrk_dxyError;
  dqm::reco::MonitorElement* htrk_dzError;
  dqm::reco::MonitorElement* htrk_qoverpError;
  dqm::reco::MonitorElement* htrk_lambdaError;
  dqm::reco::MonitorElement* htrk_phiError;
  dqm::reco::MonitorElement* htrk_dsz;
  dqm::reco::MonitorElement* htrk_dszError;
  dqm::reco::MonitorElement* htrk_vx;
  dqm::reco::MonitorElement* htrk_vy;
  dqm::reco::MonitorElement* htrk_vz;
  dqm::reco::MonitorElement* hLxy;
  dqm::reco::MonitorElement* hXError;
  dqm::reco::MonitorElement* hYError;
  dqm::reco::MonitorElement* hChi2;
  dqm::reco::MonitorElement* hZ;
  dqm::reco::MonitorElement* hx;
  dqm::reco::MonitorElement* hy;
  dqm::reco::MonitorElement* hZerror;
  dqm::reco::MonitorElement* htracksSize;
};

struct kTagProbeMuonHistos {
  kProbeKinematicMuonHistos resonanceJ_numerator;
  kProbeKinematicMuonHistos resonanceJ_denominator;
};

class ScoutingMuonTagProbeAnalyzer : public DQMGlobalEDAnalyzer<kTagProbeMuonHistos> {
public:
  explicit ScoutingMuonTagProbeAnalyzer(const edm::ParameterSet& conf);
  ~ScoutingMuonTagProbeAnalyzer() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void dqmAnalyze(const edm::Event& e, const edm::EventSetup& c, kTagProbeMuonHistos const&) const override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&, kTagProbeMuonHistos&) const override;
  void bookHistograms_resonance(DQMStore::IBooker&,
                                edm::Run const&,
                                edm::EventSetup const&,
                                kProbeKinematicMuonHistos&,
                                const std::string&) const;
  void fillHistograms_resonance(const kProbeKinematicMuonHistos histos,
                                const Run3ScoutingMuon mu,
                                const Run3ScoutingVertex vertex,
                                const float inv_mass,
                                const float lxy) const;
  bool scoutingMuonID(const Run3ScoutingMuon mu) const;

  const std::string outputInternalPath_;
  const edm::EDGetTokenT<std::vector<Run3ScoutingMuon>> scoutingMuonCollection_;
  const edm::EDGetTokenT<std::vector<Run3ScoutingVertex>> scoutingVtxCollection_;
  const Bool_t runWithoutVtx_;
};
#endif

#ifndef MuonRecoAnalyzer_H
#define MuonRecoAnalyzer_H

/** \class MuRecoAnalyzer
 *
 *  DQM monitoring source for muon reco track
 *
 *  \author G. Mila - INFN Torino
 */

#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"

class MuonRecoAnalyzer : public DQMEDAnalyzer {
public:
  /// Constructor
  MuonRecoAnalyzer(const edm::ParameterSet&);

  /// Destructor
  ~MuonRecoAnalyzer() override;

  /// Inizialize parameters for histo binning
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  //calculate residual & pull:
  void GetRes(reco::TrackRef t1, reco::TrackRef t2, std::string par, float& res, float& pull);

  //Functions needed by the SoftMuon MVA monitoring
  double getDeltaR(reco::Track track1, reco::Track track2);

  int getPv(int tidx, const reco::VertexCollection* vc);

private:
  // ----------member data ---------------------------
  edm::ParameterSet parameters;

  edm::EDGetTokenT<edm::View<reco::Muon> > theMuonCollectionLabel_;
  edm::EDGetTokenT<reco::VertexCollection> theVertexLabel_;
  edm::EDGetTokenT<reco::BeamSpot> theBeamSpotLabel_;
  edm::EDGetTokenT<DcsStatusCollection> dcsStatusCollection_;

  // Switch for verbosity
  std::string metname;
  bool doMVA;

  //histo binning parameters
  int etaBin;
  double etaMin;
  double etaMax;

  int thetaBin;
  double thetaMin;
  double thetaMax;

  int phiBin;
  double phiMin;
  double phiMax;

  int chi2Bin;
  double chi2Min;
  double chi2Max;

  int pBin;
  double pMin;
  double pMax;

  int ptBin;
  double ptMin;
  double ptMax;

  int pResBin;
  double pResMin;
  double pResMax;

  int rhBin;
  double rhMin;
  double rhMax;

  int tunePBin;
  double tunePMin;
  double tunePMax;

  //the histos
  MonitorElement* muReco;
  // global muon
  std::vector<MonitorElement*> etaGlbTrack;
  std::vector<MonitorElement*> etaResolution;
  std::vector<MonitorElement*> thetaGlbTrack;
  std::vector<MonitorElement*> thetaResolution;
  std::vector<MonitorElement*> phiGlbTrack;
  std::vector<MonitorElement*> phiResolution;
  std::vector<MonitorElement*> chi2OvDFGlbTrack;
  std::vector<MonitorElement*> probchi2GlbTrack;
  std::vector<MonitorElement*> pGlbTrack;
  std::vector<MonitorElement*> ptGlbTrack;
  std::vector<MonitorElement*> qGlbTrack;
  std::vector<MonitorElement*> qOverpResolution;
  std::vector<MonitorElement*> qOverptResolution;
  std::vector<MonitorElement*> oneOverpResolution;
  std::vector<MonitorElement*> oneOverptResolution;
  std::vector<MonitorElement*> rhAnalysis;
  std::vector<MonitorElement*> muVStkSytemRotation;
  std::vector<MonitorElement*> phiVsetaGlbTrack;
  std::vector<MonitorElement*> phiVsetaGlbTrack_badlumi;

  //Soft MVA Muon
  MonitorElement* ptSoftMuonMVA;
  MonitorElement* deltaRSoftMuonMVA;
  MonitorElement* gNchi2SoftMuonMVA;
  MonitorElement* vMuHitsSoftMuonMVA;
  MonitorElement* mNuStationsSoftMuonMVA;
  MonitorElement* dxyRefSoftMuonMVA;
  MonitorElement* dzRefSoftMuonMVA;
  MonitorElement* LWHSoftMuonMVA;
  MonitorElement* valPixHitsSoftMuonMVA;
  MonitorElement* innerChi2SoftMuonMVA;
  MonitorElement* outerChi2SoftMuonMVA;
  MonitorElement* iValFracSoftMuonMVA;
  MonitorElement* segCompSoftMuonMVA;
  MonitorElement* chi2LocMomSoftMuonMVA;
  MonitorElement* chi2LocPosSoftMuonMVA;
  MonitorElement* glbTrackTailProbSoftMuonMVA;
  MonitorElement* NTrkVHitsSoftMuonMVA;
  MonitorElement* kinkFinderSoftMuonMVA;
  MonitorElement* vRPChitsSoftMuonMVA;
  MonitorElement* glbKinkFinderSoftMuonMVA;
  MonitorElement* glbKinkFinderLogSoftMuonMVA;
  MonitorElement* staRelChi2SoftMuonMVA;
  MonitorElement* glbDeltaEtaPhiSoftMuonMVA;
  MonitorElement* trkRelChi2SoftMuonMVA;
  MonitorElement* vDThitsSoftMuonMVA;
  MonitorElement* vCSChitsSoftMuonMVA;
  MonitorElement* timeAtIpInOutSoftMuonMVA;
  MonitorElement* timeAtIpInOutErrSoftMuonMVA;
  MonitorElement* getMuonHitsPerStationSoftMuonMVA;
  MonitorElement* QprodSoftMuonMVA;

  MonitorElement* tunePResolution;

  MonitorElement* etaPull;
  MonitorElement* thetaPull;
  MonitorElement* phiPull;
  MonitorElement* qOverpPull;
  MonitorElement* qOverptPull;
  MonitorElement* oneOverpPull;
  MonitorElement* oneOverptPull;

  // tracker muon
  MonitorElement* etaTrack;
  MonitorElement* thetaTrack;
  MonitorElement* phiTrack;
  MonitorElement* chi2OvDFTrack;
  MonitorElement* probchi2Track;
  MonitorElement* pTrack;
  MonitorElement* ptTrack;
  MonitorElement* qTrack;
  // sta muon
  MonitorElement* etaStaTrack;
  MonitorElement* thetaStaTrack;
  MonitorElement* phiStaTrack;
  MonitorElement* chi2OvDFStaTrack;
  MonitorElement* probchi2StaTrack;
  MonitorElement* pStaTrack;
  MonitorElement* ptStaTrack;
  MonitorElement* qStaTrack;
  // efficiency
  std::vector<MonitorElement*> etaEfficiency;
  std::vector<MonitorElement*> phiEfficiency;

  bool IsminiAOD;
  std::string theFolder;
};
#endif

#ifndef MuonRecoAnalyzer_H
#define MuonRecoAnalyzer_H


/** \class MuRecoAnalyzer
 *
 *  DQM monitoring source for muon reco track
 *
 *  $Date: 2013/03/05 15:17:13 $
 *  $Revision: 1.10 $
 *  \author G. Mila - INFN Torino
 */


#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMOffline/Muon/src/MuonAnalyzerBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"


class MuonRecoAnalyzer : public MuonAnalyzerBase {
 public:

  /// Constructor
  MuonRecoAnalyzer(const edm::ParameterSet&, MuonServiceProxy *theService);
  
  /// Destructor
  virtual ~MuonRecoAnalyzer();

  /// Inizialize parameters for histo binning
  void beginJob(DQMStore *dbe);

  /// Get the analysis
  void analyze(const edm::Event&, const edm::EventSetup&, const reco::Muon& recoMu);

  //calculate residual & pull:
  void GetRes( reco::TrackRef t1, reco::TrackRef t2, std::string par, float &res, float &pull);

  


 private:
  // ----------member data ---------------------------
  
  edm::ParameterSet parameters;
  // Switch for verbosity
  std::string metname;
  // STA Label
  edm::InputTag theSTACollectionLabel;

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

};
#endif

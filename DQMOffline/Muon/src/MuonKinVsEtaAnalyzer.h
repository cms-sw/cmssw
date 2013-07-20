#ifndef MuonKinVsEtaAnalyzer_H
#define MuonKinVsEtaAnalyzer_H


/** \class MuRecoAnalyzer
 *
 *  DQM monitoring source for muon reco track
 *
 *  $Date: 2012/08/27 11:46:54 $
 *  $Revision: 1.6 $
 *  \author S. Goy Lopez, CIEMAT
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


class MuonKinVsEtaAnalyzer : public MuonAnalyzerBase {
 public:

  /// Constructor
  MuonKinVsEtaAnalyzer(const edm::ParameterSet&, MuonServiceProxy *theService);
  
  /// Destructor
  virtual ~MuonKinVsEtaAnalyzer();

  /// Iniyeszialize parameters for histo binning
  void beginJob(DQMStore *dbe);

  /// Get the analysis
  void analyze(const edm::Event&, const edm::EventSetup&, const reco::Muon& recoMu);


 private:
  // ----------member data ---------------------------
  
  edm::ParameterSet parameters;
  // Switch for verbosity
  std::string metname;
  // STA Label
  edm::InputTag theSTACollectionLabel;

  //Vertex requirements
  edm::InputTag  vertexTag;
  edm::InputTag  bsTag;


  //histo binning parameters
  int pBin;
  double pMin;
  double pMax;

  int ptBin;
  double ptMin;
  double ptMax;

  int etaBin;
  double etaMin;
  double etaMax;

  int phiBin;
  double phiMin;
  double phiMax;

  int chiBin;
  double chiMin;
  double chiMax;

  double chiprobMin;
  double chiprobMax;

  //Defining relevant eta regions
  double EtaCutMin;
  double EtaCutMax;
  double etaBMin;
  double etaBMax;
  double etaECMin;
  double etaECMax;
  double etaOvlpMin;
  double etaOvlpMax;

  //the histos
  // global muon
  std::vector<MonitorElement*> etaGlbTrack;
  std::vector<MonitorElement*> phiGlbTrack;
  std::vector<MonitorElement*> pGlbTrack;
  std::vector<MonitorElement*> ptGlbTrack;
  std::vector<MonitorElement*> chi2GlbTrack;
  std::vector<MonitorElement*> chi2probGlbTrack;

  // tracker muon
  std::vector<MonitorElement*> etaTrack;
  std::vector<MonitorElement*> phiTrack;
  std::vector<MonitorElement*> pTrack;
  std::vector<MonitorElement*> ptTrack;
  std::vector<MonitorElement*> chi2Track;
  std::vector<MonitorElement*> chi2probTrack;

  // sta muon
  std::vector<MonitorElement*> etaStaTrack;
  std::vector<MonitorElement*> phiStaTrack;
  std::vector<MonitorElement*> pStaTrack;
  std::vector<MonitorElement*> ptStaTrack;
  std::vector<MonitorElement*> chi2StaTrack;
  std::vector<MonitorElement*> chi2probStaTrack;

  // GMPT muon
  std::vector<MonitorElement*> etaTightTrack;
  std::vector<MonitorElement*> phiTightTrack;
  std::vector<MonitorElement*> pTightTrack;
  std::vector<MonitorElement*> ptTightTrack;
  std::vector<MonitorElement*> chi2TightTrack;
  std::vector<MonitorElement*> chi2probTightTrack;

  // Loose muon;
  std::vector<MonitorElement*> etaLooseTrack;
  std::vector<MonitorElement*> phiLooseTrack;
  std::vector<MonitorElement*> pLooseTrack;
  std::vector<MonitorElement*> ptLooseTrack;
  std::vector<MonitorElement*> chi2LooseTrack;
  std::vector<MonitorElement*> chi2probLooseTrack;

  // Soft muon;
  std::vector<MonitorElement*> etaSoftTrack;
  std::vector<MonitorElement*> phiSoftTrack;
  std::vector<MonitorElement*> pSoftTrack;
  std::vector<MonitorElement*> ptSoftTrack;
  std::vector<MonitorElement*> chi2SoftTrack;
  std::vector<MonitorElement*> chi2probSoftTrack;

 // HighPt muon;
  std::vector<MonitorElement*> etaHighPtTrack;
  std::vector<MonitorElement*> phiHighPtTrack;
  std::vector<MonitorElement*> pHighPtTrack;
  std::vector<MonitorElement*> ptHighPtTrack;
  std::vector<MonitorElement*> chi2HighPtTrack;
  std::vector<MonitorElement*> chi2probHighPtTrack;

};
#endif

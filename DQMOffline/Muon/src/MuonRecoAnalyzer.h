#ifndef MuonRecoAnalyzer_H
#define MuonRecoAnalyzer_H


/** \class MuRecoAnalyzer
 *
 *  DQM monitoring source for muon reco track
 *
 *  $Date: 2008/04/02 12:18:41 $
 *  $Revision: 1.1 $
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
  void beginJob(edm::EventSetup const& iSetup, DQMStore *dbe);

  /// Get the analysis
  void analyze(const edm::Event&, const edm::EventSetup&, const reco::Muon& recoMu);


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

  int pBin;
  double pMin;
  double pMax;

  int ptBin;
  double ptMin;
  double ptMax;

  int pResBin;
  double pResMin;
  double pResMax;


  //the histos
  MonitorElement* muReco;
  // global muon
  std::vector<MonitorElement*> etaGlbTrack;
  std::vector<MonitorElement*> etaResolution;
  std::vector<MonitorElement*> thetaGlbTrack;
  std::vector<MonitorElement*> thetaResolution;
  std::vector<MonitorElement*> phiGlbTrack;
  std::vector<MonitorElement*> phiResolution;
  std::vector<MonitorElement*> pGlbTrack;
  std::vector<MonitorElement*> ptGlbTrack;
  std::vector<MonitorElement*> qGlbTrack;
  std::vector<MonitorElement*> qOverpResolution;
  std::vector<MonitorElement*> qOverptResolution;
  std::vector<MonitorElement*> oneOverpResolution;
  std::vector<MonitorElement*> oneOverptResolution;
  // tracker muon
  MonitorElement* etaTrack;
  MonitorElement* thetaTrack;
  MonitorElement* phiTrack;
  MonitorElement* pTrack;
  MonitorElement* ptTrack;
  MonitorElement* qTrack;
  // sta muon
  MonitorElement* etaStaTrack;
  MonitorElement* thetaStaTrack;
  MonitorElement* phiStaTrack;
  MonitorElement* pStaTrack;
  MonitorElement* ptStaTrack;
  MonitorElement* qStaTrack;

};
#endif

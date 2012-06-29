#ifndef MuonKinVsEtaAnalyzer_H
#define MuonKinVsEtaAnalyzer_H


/** \class MuRecoAnalyzer
 *
 *  DQM monitoring source for muon reco track
 *
 *  $Date: 2011/07/14 13:27:43 $
 *  $Revision: 1.3 $
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

  /// Inizialize parameters for histo binning
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

  // tracker muon
  std::vector<MonitorElement*> etaTrack;
  std::vector<MonitorElement*> phiTrack;
  std::vector<MonitorElement*> pTrack;
  std::vector<MonitorElement*> ptTrack;

  // sta muon
  std::vector<MonitorElement*> etaStaTrack;
  std::vector<MonitorElement*> phiStaTrack;
  std::vector<MonitorElement*> pStaTrack;
  std::vector<MonitorElement*> ptStaTrack;

  // GMPT muon
  std::vector<MonitorElement*> etaTightTrack;
  std::vector<MonitorElement*> phiTightTrack;
  std::vector<MonitorElement*> pTightTrack;
  std::vector<MonitorElement*> ptTightTrack;
  std::vector<MonitorElement*> chi2TightTrack;
  std::vector<MonitorElement*> chi2probTightTrack;

};
#endif

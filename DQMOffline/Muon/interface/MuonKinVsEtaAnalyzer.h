#ifndef MuonKinVsEtaAnalyzer_H
#define MuonKinVsEtaAnalyzer_H
/** \class MuRecoAnalyzer
 *
 *  DQM monitoring source for muon reco track
 *
 *  \author S. Goy Lopez, CIEMAT
 *  \author S. Folgueras, U. Oviedo
 */

#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h" 
#include "DataFormats/MuonReco/interface/MuonEnergy.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

class MuonKinVsEtaAnalyzer : public DQMEDAnalyzer {
 public:
  
  /// Constructor
  MuonKinVsEtaAnalyzer(const edm::ParameterSet& pSet);
  
  /// Destructor
  ~MuonKinVsEtaAnalyzer();
  
  void analyze(const edm::Event&, const edm::EventSetup&);
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

 private:
  
  // ----------member data ---------------------------
  MuonServiceProxy *theService;
  DQMStore* theDbe;
  edm::ParameterSet parameters;
 
  // Switch for verbosity
  std::string metname;

  //Vertex requirements
  edm::EDGetTokenT<reco::MuonCollection>   theMuonCollectionLabel_;
  edm::EDGetTokenT<reco::VertexCollection> theVertexLabel_;
  edm::EDGetTokenT<reco::BeamSpot>         theBeamSpotLabel_;
  
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

  // Medium muon;
  std::vector<MonitorElement*> etaMediumTrack;
  std::vector<MonitorElement*> phiMediumTrack;
  std::vector<MonitorElement*> pMediumTrack;
  std::vector<MonitorElement*> ptMediumTrack;
  std::vector<MonitorElement*> chi2MediumTrack;
  std::vector<MonitorElement*> chi2probMediumTrack;

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

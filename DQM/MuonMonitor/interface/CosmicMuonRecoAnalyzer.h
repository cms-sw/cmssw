#ifndef CosmicMuonRecoAnalyzer_H
#define CosmicMuonRecoAnalyzer_H

/** \class CosmicMuRecoAnalyzer
 *
 *  DQM monitoring source for muon reco track
 *
 *  \author A. Calderon  IFCA-CSIC-Unv.Cantabria
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
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h" 
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"


class CosmicMuonRecoAnalyzer : public DQMEDAnalyzer {
 public:

  /// Constructor
  CosmicMuonRecoAnalyzer(const edm::ParameterSet&);
  
  /// Destructor
  ~CosmicMuonRecoAnalyzer() override;

  /// Inizialize parameters for histo binning
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
 
 private:
  // ----------member data ---------------------------
  MuonServiceProxy *theService;
  edm::ParameterSet parameters;
 
  edm::EDGetTokenT<edm::View<reco::Track> >   theMuonCollectionLabel_;

   //histo binning parameters

  int  hitsBin;
  int hitsMin;
  int  hitsMax;

   int nTrkBin;
  int nTrkMax;
    int nTrkMin;

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


  // sta muon
  MonitorElement* nTracksSta;

  MonitorElement* etaStaTrack;
  MonitorElement* thetaStaTrack;
  MonitorElement* phiStaTrack;
  MonitorElement* chi2OvDFStaTrack;
  MonitorElement* probchi2StaTrack;
  MonitorElement* pStaTrack;
  MonitorElement* qOverPStaTrack;
  MonitorElement* ptStaTrack;
  MonitorElement* qStaTrack;
  MonitorElement* nValidHitsStaTrack;

  MonitorElement* qOverPStaTrack_p;
  MonitorElement* phiVsetaStaTrack;
  MonitorElement* nValidHitsStaTrack_eta;
  MonitorElement* nValidHitsStaTrack_phi;


  std::string theFolder;
};
#endif

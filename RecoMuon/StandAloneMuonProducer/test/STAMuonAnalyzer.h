#ifndef RecoMuon_StandAloneMuonProducer_STAMuonAnalyzer_H
#define RecoMuon_StandAloneMuonProducer_STAMuonAnalyzer_H

/** \class STAMuonAnalyzer
 *  Analyzer of the StandAlone muon tracks
 *
 *  $Date: 2009/10/31 05:19:45 $
 *  $Revision: 1.3 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

// Base Class Headers
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class TFile;
class TH1F;
class TH2F;

class STAMuonAnalyzer: public edm::EDAnalyzer {
public:
  /// Constructor
  STAMuonAnalyzer(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~STAMuonAnalyzer();

  // Operations

  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);

  virtual void beginJob() ;
  virtual void endJob() ;

protected:

private:
  std::string theRootFileName;
  TFile* theFile;

  std::map<std::string,TH1F*> histContainer_;
  std::map<std::string,TH2F*> histContainer2D_; 

  edm::InputTag staTrackLabel_;
  std::string theSeedCollectionLabel;
  bool noGEMCase_;
  bool isGlobalMuon_;
  bool includeME11_;

  // Histograms
  TH1F *hPtRec;
  TH1F *hDeltaPtRec;
  TH1F *hPtSim; 
  TH1F *hPres;
  TH1F *h1_Pres;
  TH1F *h1_PresMuon;
  TH1F *h1_PresMuon2;
  TH1F *hPTDiff;
  TH1F *hPTDiff2;
  TH2F *hPTDiffvsEta;
  TH2F *hPTDiffvsPhi;
  TH1F *hSimEta;
  TH1F *hRecEta;
  TH1F *hDeltaEta;
  TH1F *hDeltaPhi;
  TH1F *hDeltaPhiMuon;
  TH1F *hDeltaPhiPlus;
  TH1F *hDeltaPhiMinus;
  TH1F *hSimPhi;
  TH1F *hRecPhi;
  TH1F *hNumSimTracks;
  TH1F *hNumMuonSimTracks;
  TH1F *hNumRecTracks;
  TH1F *hNumGEMSimHits;
  TH1F *hNumCSCSimHits;
  TH1F *hNumGEMRecHits;
  TH1F *hNumGEMRecHitsMuon;
  TH1F *hNumCSCRecHits;
  TH2F *hPtResVsPt;
  TH2F *hInvPtResVsPt;
  TH2F *hInvPtResVsPtMuon;
  TH2F *hPtResVsEta;
  TH2F *hInvPtResVsEta;
  TH2F *hInvPtResVsEtaMuon;
  TH2F *hInvPtResVsPtSel;
  TH2F *hPtResVsPtNoCharge;
  TH2F *hInvPtResVsPtNoCharge;
  TH2F *hPtResVsEtaNoCharge;
  TH2F *hInvPtResVsEtaNoCharge;
  TH2F *hDPhiVsPt;
  TH1F *hDenPt;
  TH1F *hDenEta;
  TH1F *hDenPhi;
  TH1F *hDenPhiPlus;
  TH1F *hDenPhiMinus;
  TH1F *hDenSimPt;
  TH1F *hDenSimEta;
  TH1F *hDenSimPhiPlus;
  TH1F *hDenSimPhiMinus;
  TH1F *hNumPt;
  TH1F *hNumEta;
  TH1F *hNumPhi;
  TH1F *hNumPhiPlus;
  TH1F *hNumPhiMinus;
  TH1F *hNumSimPt;
  TH1F *hNumSimEta;
  TH1F *hNumSimPhiPlus;
  TH1F *hNumSimPhiMinus;
  TH1F *hPullGEMx;
  TH1F *hPullGEMy;
  TH1F *hPullGEMz;
  TH1F *hPullCSC;
  TH1F *hGEMRecHitEta;
  TH1F *hGEMRecHitPhi;
  TH1F *hDR;
  TH1F *hDR2;
  TH1F *hDR3;
  TH2F *hRecPhi2DPlusLayer1;
  TH2F *hRecPhi2DMinusLayer1;  
  TH2F *hRecPhi2DPlusLayer2;
  TH2F *hRecPhi2DMinusLayer2;  
  TH2F *hDeltaCharge;
  TH2F *hDeltaChargeMuon;
  TH2F *hDeltaChargeVsEta;
  TH2F *hDeltaChargeVsEtaMuon;
  TH2F *hCharge;
  TH2F *hDeltaPhiVsSimTrackPhi;
  TH2F *hDeltaPhiVsSimTrackPhi2;
  TH2F *hRecoPtVsSimPt;
  TH2F *hDeltaPtVsSimPt;
  TH1F *hTracksWithME11;
  TH1F *hPtSimCorr;
  TH2F *hPtResVsPtCorr;
  TH2F *hInvPtResVsPtCorr;
  TH1F *hCSCorGEM;
  TH1F *hSimTrackMatch;
  TH1F *hRecHitMatching;
  TH1F *hRecHitParMatching;
  TH2F *hDRMatchVsPt;
  TH2F *hDRMatchVsPtMuon;
  TH1F *hMatchedSimHits;
  TH2F *hRecoTracksWithMatchedRecHits;
  TH2F *hDeltaQvsDeltaPt;
  TH2F *hCheckGlobalTracksVsPt;
  TH2F *hCheckTracksVsPt;
  TH2F *hPtResVsPtRes;
  TH1F *hDeltaPtRes;
  TH1F *hCountPresence;

  // Counters
  int numberOfSimTracks;
  int numberOfRecTracks;

  std::string theDataType;
  
};
#endif


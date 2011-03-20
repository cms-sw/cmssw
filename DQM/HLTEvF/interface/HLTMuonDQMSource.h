#ifndef HLTMuonDQMSource_H
#define HLTMuonDQMSource_H

/** \class HLTMuonDQMSource
 * *
 *  DQM Test Client
 *
 *  $Date: 2010/04/27 23:26:25 $
 *  $Revision: 1.20 $
 *  \author  M. Vander Donckt CERN
 *   
 */
#include <memory>
#include <unistd.h>
#include <vector>
#include <string>

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//
// class declaration
//

class HLTMuonDQMSource : public edm::EDAnalyzer {
public:
  HLTMuonDQMSource( const edm::ParameterSet& );
  ~HLTMuonDQMSource();

protected:
   
  /// BeginJob
  void beginJob();

  /// BeginRun
  void beginRun(const edm::Run& run, const edm::EventSetup& eventSetup);

  /// Fake Analyze
  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) ;

  void beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock, 
                            const edm::EventSetup& eventSetup) ;

  /// DQM Client Diagnostic
  void endLuminosityBlock(const edm::LuminosityBlock& lumiBlock, 
                          const edm::EventSetup& eventSetup);

  /// EndRun
  void endRun(const edm::Run& run, const edm::EventSetup& eventSetup);

  /// Endjob
  void endJob();

private:
 
  edm::ParameterSet parameters_;

  DQMStore* dbe_;  
  std::string monitorName_;
  std::string outputFile_;
  int counterEvt_;      ///counter
  int nTrig_;		/// mutriggered events
  int prescaleEvt_;     ///every n events
  double coneSize_;
  int nl3muon, nl3muonOIState, nl3muonOIHit, nl3muonIOHit;
  edm::InputTag l2collectionTag_;
  edm::InputTag l2isolationTag_;
  edm::InputTag l2seedscollectionTag_;
  edm::InputTag l3seedscollectionTag_;
  edm::InputTag l3collectionTag_;
  edm::InputTag l3isolationTag_;
  //  edm::InputTag l3linksTag_;
  edm::InputTag TrigResultInput;

  std::vector<std::string> theTriggerBits;
  std::vector<std::string> theDirectoryName;
  std::vector<std::string> theHLTCollectionLevel;
  std::string striggers_[20];
 
  edm::InputTag l3seedscollectionTagOIState_;
  edm::InputTag l3seedscollectionTagOIHit_;
  edm::InputTag l3seedscollectionTagIOHit_;

  edm::InputTag l3trkfindingOIState_;
  edm::InputTag l3trkfindingOIHit_;
  edm::InputTag l3trkfindingIOHit_;

  edm::InputTag l3trkOIState_;
  edm::InputTag l3trkOIHit_;
  edm::InputTag l3trkIOHit_;
  edm::InputTag l3tktrk_;

  edm::InputTag l3muons_;
  edm::InputTag l3muonsOIState_;
  edm::InputTag l3muonsOIHit_;
  edm::InputTag l3muonsIOHit_;
  // ----------member data ---------------------------
  bool verbose_;
  static const int NTRIG = 20;
  int nTrigs;

  MonitorElement * hNMu[NTRIG][5];
  MonitorElement * hcharge[NTRIG][5];
  MonitorElement * hchargeconv[NTRIG][3];
  MonitorElement * hpt[NTRIG][5];
  MonitorElement * heta[NTRIG][5];
  MonitorElement * hphi[NTRIG][5];
  MonitorElement * hphi_norm[NTRIG][5];
  MonitorElement * hptphi[NTRIG][5];
  MonitorElement * hpteta[NTRIG][5];
  MonitorElement * hptres[NTRIG][3];
  MonitorElement * hptrespt[NTRIG][3];
  MonitorElement * hetares[NTRIG][3];
  MonitorElement * hetareseta[NTRIG][3];
  MonitorElement * hphires[NTRIG][3];
  MonitorElement * hphiresphi[NTRIG][3];
  MonitorElement * hetaphi[NTRIG][5];
  MonitorElement * hdr[NTRIG][2];
  MonitorElement * hd0[NTRIG][2];
  MonitorElement * hdz[NTRIG][2];
  MonitorElement * hdz0[NTRIG][2];
  MonitorElement * hdrphi[NTRIG][2];
  MonitorElement * hd0phi[NTRIG][2];
  MonitorElement * hdzeta[NTRIG][2];
  MonitorElement * hdz0eta[NTRIG][2];
  MonitorElement * herr0[NTRIG][2];
  MonitorElement * hnHits[NTRIG][4];
  MonitorElement * hnValidHits[NTRIG];
  MonitorElement * hnTkValidHits[NTRIG];
  MonitorElement * hnPixelValidHits[NTRIG];
  MonitorElement * hnStripValidHits[NTRIG];
  MonitorElement * hnMuValidHits[NTRIG];
  MonitorElement * hdimumass[NTRIG][2];
  MonitorElement * hisoL2[NTRIG];
  MonitorElement * hisoL3[NTRIG];
  MonitorElement * hl1quality[NTRIG];
  MonitorElement * hptfrac[NTRIG][2];
  MonitorElement * hetafrac[NTRIG][2];
  MonitorElement * hphifrac[NTRIG][2];
  MonitorElement * hseedptres[NTRIG][2];
  MonitorElement * hseedetares[NTRIG][2];
  MonitorElement * hseedphires[NTRIG][2];
  MonitorElement * hptpull[NTRIG];
  MonitorElement * hptpullpt[NTRIG];
  MonitorElement * hetapull[NTRIG];
  MonitorElement * hetapulleta[NTRIG];
  MonitorElement * hphipull[NTRIG];
  MonitorElement * hphipullphi[NTRIG];
  MonitorElement * hseedNMuper[NTRIG][2];
  MonitorElement * hptrelres[NTRIG][3];
  MonitorElement * hptrelrespt[NTRIG][3];
  MonitorElement * hetarelres[NTRIG][3];
  MonitorElement * hetarelreseta[NTRIG][3];
  MonitorElement * hphirelres[NTRIG][3];
  MonitorElement * hphirelresphi[NTRIG][3];
  MonitorElement * hseedptrelres[NTRIG][2];
  MonitorElement * hseedetarelres[NTRIG][2];
  MonitorElement * hseedphirelres[NTRIG][2];
  MonitorElement * htowerEt[NTRIG];
  MonitorElement * htowerEta[NTRIG];
  MonitorElement * htowerPhi[NTRIG];
  MonitorElement * htowerHadEnergy[NTRIG];
  MonitorElement * htowerEmEnergy[NTRIG];
  MonitorElement * htowerOuterEnergy[NTRIG];
  MonitorElement * htowerHadEt[NTRIG];
  MonitorElement * htowerEmEt[NTRIG];
  MonitorElement * htowerOuterEt[NTRIG];
  MonitorElement * htowerEtaHadEt[NTRIG];
  MonitorElement * htowerEtaEmEt[NTRIG];
  MonitorElement * htowerEtaOuterEt[NTRIG];
  MonitorElement * htowerPhiHadEt[NTRIG];
  MonitorElement * htowerPhiEmEt[NTRIG];
  MonitorElement * htowerPhiOuterEt[NTRIG];
  MonitorElement * htowerdRL2[NTRIG];
  MonitorElement * htowerdRL3[NTRIG];
  MonitorElement * hL2muonIsoDR[NTRIG];
  MonitorElement * hL3muonIsoDR[NTRIG];

  float XMIN; float XMAX;

  TH1D *_hpt1[NTRIG][2], *_hpt2[NTRIG][2];
  TH1D *_heta1[NTRIG][2], *_heta2[NTRIG][2];
  TH1D *_hphi1[NTRIG][2], *_hphi2[NTRIG][2];

/// added plots
  MonitorElement * hd0_OIState[NTRIG][5];
  MonitorElement * hd0_OIHit[NTRIG][5];
  MonitorElement * hd0_IOHit[NTRIG][5];
  MonitorElement * hdz_OIState[NTRIG][5];
  MonitorElement * hdz_OIHit[NTRIG][5];
  MonitorElement * hdz_IOHit[NTRIG][5];
  MonitorElement * hdr_OIState[NTRIG][5];
  MonitorElement * hdr_OIHit[NTRIG][5];
  MonitorElement * hdr_IOHit[NTRIG][5];

  MonitorElement * hNMu_comp[NTRIG][5];
  MonitorElement * hNMu_trk_comp[NTRIG][5];
  MonitorElement * hNMu_l3seed_comp[NTRIG][5];

// OIState,OIHit,IOHit
  MonitorElement * hNMu_OIState[NTRIG][5];
  MonitorElement * hNMu_OIHit[NTRIG][5];
  MonitorElement * hNMu_IOHit[NTRIG][5];
  MonitorElement * hpt_OIState[NTRIG][5];
  MonitorElement * hpt_OIHit[NTRIG][5];
  MonitorElement * hpt_IOHit[NTRIG][5];
  MonitorElement * heta_OIState[NTRIG][5];
  MonitorElement * heta_OIHit[NTRIG][5];
  MonitorElement * heta_IOHit[NTRIG][5];
  MonitorElement * hphi_OIState[NTRIG][5];
  MonitorElement * hphi_OIHit[NTRIG][5];
  MonitorElement * hphi_IOHit[NTRIG][5];
  MonitorElement * hpteta_OIState[NTRIG][5];
  MonitorElement * hpteta_OIHit[NTRIG][5];
  MonitorElement * hpteta_IOHit[NTRIG][5];
  MonitorElement * hetaphi_OIState[NTRIG][5];
  MonitorElement * hetaphi_OIHit[NTRIG][5];
  MonitorElement * hetaphi_IOHit[NTRIG][5];
  MonitorElement * hptphi_OIState[NTRIG][5];
  MonitorElement * hptphi_OIHit[NTRIG][5];
  MonitorElement * hptphi_IOHit[NTRIG][5];
  MonitorElement * hcharge_OIState[NTRIG][5];
  MonitorElement * hcharge_OIHit[NTRIG][5];
  MonitorElement * hcharge_IOHit[NTRIG][5];
// tracker
  MonitorElement * hNMu_trk[NTRIG][5];
  MonitorElement * hcharge_trk[NTRIG][5];
  MonitorElement * hpt_trk[NTRIG][5];
  MonitorElement * heta_trk[NTRIG][5];
  MonitorElement * hphi_trk[NTRIG][5];
  MonitorElement * hpteta_trk[NTRIG][5];
  MonitorElement * hptphi_trk[NTRIG][5];
  MonitorElement * hetaphi_trk[NTRIG][5];
  MonitorElement * hd0_trk[NTRIG][5];
  MonitorElement * hdz_trk[NTRIG][5];
  MonitorElement * hdr_trk[NTRIG][5];
  MonitorElement * hNMu_trk_OIState[NTRIG][5];
  MonitorElement * hcharge_trk_OIState[NTRIG][5];
  MonitorElement * hpt_trk_OIState[NTRIG][5];
  MonitorElement * heta_trk_OIState[NTRIG][5];
  MonitorElement * hphi_trk_OIState[NTRIG][5];
  MonitorElement * hpteta_trk_OIState[NTRIG][5];
  MonitorElement * hptphi_trk_OIState[NTRIG][5];
  MonitorElement * hetaphi_trk_OIState[NTRIG][5];
  MonitorElement * hd0_trk_OIState[NTRIG][5];
  MonitorElement * hdz_trk_OIState[NTRIG][5];
  MonitorElement * hdr_trk_OIState[NTRIG][5];
  MonitorElement * hNMu_trk_OIHit[NTRIG][5];
  MonitorElement * hcharge_trk_OIHit[NTRIG][5];
  MonitorElement * hpt_trk_OIHit[NTRIG][5];
  MonitorElement * heta_trk_OIHit[NTRIG][5];
  MonitorElement * hphi_trk_OIHit[NTRIG][5];
  MonitorElement * hpteta_trk_OIHit[NTRIG][5];
  MonitorElement * hptphi_trk_OIHit[NTRIG][5];
  MonitorElement * hetaphi_trk_OIHit[NTRIG][5];
  MonitorElement * hd0_trk_OIHit[NTRIG][5];
  MonitorElement * hdz_trk_OIHit[NTRIG][5];
  MonitorElement * hdr_trk_OIHit[NTRIG][5];
  MonitorElement * hNMu_trk_IOHit[NTRIG][5];
  MonitorElement * hcharge_trk_IOHit[NTRIG][5];
  MonitorElement * hpt_trk_IOHit[NTRIG][5];
  MonitorElement * heta_trk_IOHit[NTRIG][5];
  MonitorElement * hphi_trk_IOHit[NTRIG][5];
  MonitorElement * hpteta_trk_IOHit[NTRIG][5];
  MonitorElement * hptphi_trk_IOHit[NTRIG][5];
  MonitorElement * hetaphi_trk_IOHit[NTRIG][5];
  MonitorElement * hd0_trk_IOHit[NTRIG][5];
  MonitorElement * hdz_trk_IOHit[NTRIG][5];
  MonitorElement * hdr_trk_IOHit[NTRIG][5];

  MonitorElement * hptres_L3L3trk[NTRIG][5];
  MonitorElement * hetares_L3L3trk[NTRIG][5];
  MonitorElement * hphires_L3L3trk[NTRIG][5];
  MonitorElement * hptrelres_L3L3trk[NTRIG][5];
  MonitorElement * hetarelres_L3L3trk[NTRIG][5];
  MonitorElement * hphirelres_L3L3trk[NTRIG][5];
};

#endif


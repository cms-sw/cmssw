#ifndef MuonAnalyzer_H
#define MuonAnalyzer_H


/** \class MuonAnalyzer
 *
 *  DQM muon analysis monitoring
 *
 *  $Date: 2010/07/19 22:17:37 $
 *  $Revision: 1.16 $
 *  \author G. Mila - INFN Torino
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MuonEnergyDepositAnalyzer;
class MuonSeedsAnalyzer;
class MuonRecoAnalyzer;
class MuonKinVsEtaAnalyzer;
class SegmentTrackAnalyzer;
class MuonKinVsEtaAnalyzer;
class DiMuonHistograms;
class DQMStore;
class MuonServiceProxy;

class MuonAnalyzer : public edm::EDAnalyzer {
 public:

  /// Constructor
  MuonAnalyzer(const edm::ParameterSet&);
  
  /// Destructor
  virtual ~MuonAnalyzer();
  
  /// Inizialize parameters for histo binning
  void beginJob(void);

  /// Get the analysis
  void analyze(const edm::Event&, const edm::EventSetup&);

  /// Save the histos
  void endJob(void);

 private:
  // ----------member data ---------------------------
  
  DQMStore* theDbe;
  edm::ParameterSet parameters;
  MuonServiceProxy *theService;
  // Switch for verbosity
  std::string metname;

  // Muon Label
  edm::InputTag theMuonCollectionLabel;
  // Glb Muon Track label
  edm::InputTag theGlbMuTrackCollectionLabel;
  // Sta Muon Track label
  edm::InputTag theStaMuTrackCollectionLabel;
  // Seed Label
  edm::InputTag theSeedsCollectionLabel;
  
  bool theMuEnergyAnalyzerFlag;
  bool theSeedsAnalyzerFlag;
  bool theMuonRecoAnalyzerFlag;
  bool theMuonKinVsEtaAnalyzerFlag;
  bool theMuonSegmentsAnalyzerFlag;
  bool theDiMuonHistogramsFlag;
  // the muon energy analyzer
  MuonEnergyDepositAnalyzer * theMuEnergyAnalyzer;
  // the seeds analyzer
  MuonSeedsAnalyzer * theSeedsAnalyzer;
  // the muon reco analyzer
  MuonRecoAnalyzer * theMuonRecoAnalyzer;
  // the muon kin vs eta analyzer
  MuonKinVsEtaAnalyzer * theMuonKinVsEtaAnalyzer;
  // the track segments analyzer for glb muons
  SegmentTrackAnalyzer * theGlbMuonSegmentsAnalyzer;
  // the track segments analyzer for sta muons
  SegmentTrackAnalyzer * theStaMuonSegmentsAnalyzer;
  // The dimuon histograms
  DiMuonHistograms * theDiMuonHistograms;
  
};
#endif  

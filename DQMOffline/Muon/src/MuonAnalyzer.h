#ifndef MuonAnalyzer_H
#define MuonAnalyzer_H


/** \class MuonAnalyzer
 *
 *  DQM muon analysis monitoring
 *
 *  $Date: 2011/05/22 18:17:21 $
 *  $Revision: 1.17 $
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
class MuonRecoOneHLT;

class MuonAnalyzer : public edm::EDAnalyzer {
 public:

  /// Constructor
  MuonAnalyzer(const edm::ParameterSet&);
  
  /// Destructor
  virtual ~MuonAnalyzer();
  
  /// Inizialize parameters for histo binning
  void beginJob(void);
  void beginRun(const edm::Run&, const edm::EventSetup&);

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
  bool theMuonRecoOneHLTAnalyzerFlag;
  
  // Define Analysis Modules
  MuonEnergyDepositAnalyzer* theMuEnergyAnalyzer;
  MuonSeedsAnalyzer*         theSeedsAnalyzer;
  MuonRecoAnalyzer*          theMuonRecoAnalyzer;
  MuonKinVsEtaAnalyzer*      theMuonKinVsEtaAnalyzer;
  SegmentTrackAnalyzer*      theGlbMuonSegmentsAnalyzer;
  SegmentTrackAnalyzer*      theStaMuonSegmentsAnalyzer;
  DiMuonHistograms*          theDiMuonHistograms;
  MuonRecoOneHLT*            theMuonRecoOneHLTAnalyzer;
};
#endif  

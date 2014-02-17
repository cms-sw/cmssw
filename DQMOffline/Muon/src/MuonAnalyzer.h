#ifndef MuonAnalyzer_H
#define MuonAnalyzer_H


/** \class MuonAnalyzer
 *
 *  DQM muon analysis monitoring
 *
 *  $Date: 2011/11/01 11:40:13 $
 *  $Revision: 1.19 $
 *  \author G. Mila - INFN Torino
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
//
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"

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
class EfficiencyAnalyzer;

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
  edm::InputTag theTriggerResultsLabel;

  bool theMuEnergyAnalyzerFlag;
  bool theSeedsAnalyzerFlag;
  bool theMuonRecoAnalyzerFlag;
  bool theMuonKinVsEtaAnalyzerFlag;
  bool theMuonSegmentsAnalyzerFlag;
  bool theDiMuonHistogramsFlag;
  bool theMuonRecoOneHLTAnalyzerFlag;
  bool theEfficiencyAnalyzerFlag;
 
  // Define Analysis Modules
  MuonEnergyDepositAnalyzer* theMuEnergyAnalyzer;
  MuonSeedsAnalyzer*         theSeedsAnalyzer;
  MuonRecoAnalyzer*          theMuonRecoAnalyzer;
  MuonKinVsEtaAnalyzer*      theMuonKinVsEtaAnalyzer;
  SegmentTrackAnalyzer*      theGlbMuonSegmentsAnalyzer;
  SegmentTrackAnalyzer*      theStaMuonSegmentsAnalyzer;
  DiMuonHistograms*          theDiMuonHistograms;
  MuonRecoOneHLT*            theMuonRecoOneHLTAnalyzer;
  EfficiencyAnalyzer*        theEfficiencyAnalyzer; 
};
#endif  

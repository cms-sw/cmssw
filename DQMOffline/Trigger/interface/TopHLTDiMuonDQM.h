#ifndef TopHLTDiMuonDQM_H
#define TopHLTDiMuonDQM_H

/*
 *  DQM HLT Dimuon Test Client
 *
 *  $Date: 2010/01/12 16:29:21 $
 *  $Revision: 1.2 $
 *  \author  M. Vander Donckt CERN
 *   
 */

#include <memory>
#include <string>
#include <functional>

#include <unistd.h>
#include <stdlib.h>
#include <string.h>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include <DataFormats/Common/interface/Ref.h>

#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeedCollection.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeedCollection.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "HLTrigger/HLTfilters/interface/HLTHighLevel.h"

//
// class declaration
//

class TopHLTDiMuonDQM : public edm::EDAnalyzer {

 public:

  TopHLTDiMuonDQM( const edm::ParameterSet& );
  ~TopHLTDiMuonDQM();

 protected:   

  void beginJob(void);
  void beginRun(const edm::Run& r, const edm::EventSetup& c);
  void analyze(const edm::Event& e, const edm::EventSetup& c);
  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context);
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c);
  void endRun(const edm::Run& r, const edm::EventSetup& c);
  void endJob();

 private:

  edm::ParameterSet parameters_;
  DQMStore* dbe_;
  std::string monitorName_;
  std::string level_;
  int counterEvt_;
  int prescaleEvt_;

  int N_sig[100];
  int N_trig[100];
  float Eff[100];

  edm::InputTag triggerResults_;
  edm::InputTag L1_Collection_;
  edm::InputTag L3_Collection_;
  edm::InputTag L3_Isolation_;

  std::vector<std::string> hltPaths_L1_;
  std::vector<std::string> hltPaths_L3_;
  std::vector<std::string> hltPaths_sig_;
  std::vector<std::string> hltPaths_trig_;

  double muon_pT_cut_;
  double muon_eta_cut_;
  double MassWindow_up_;
  double MassWindow_down_;

  // ----------member data ---------------------------

  bool verbose_;

  MonitorElement * Trigs;
  MonitorElement * NMuons;
  MonitorElement * NMuons_charge;
  MonitorElement * NMuons_iso;
  MonitorElement * PtMuons;
  MonitorElement * PtMuons_sig;
  MonitorElement * PtMuons_trig;
  MonitorElement * EtaMuons;
  MonitorElement * EtaMuons_sig;
  MonitorElement * EtaMuons_trig;
  MonitorElement * PhiMuons;

  MonitorElement * DiMuonMassRC;
  MonitorElement * DiMuonMassWC;
  MonitorElement * DiMuonMassRC_LOGX;
  MonitorElement * DiMuonMassWC_LOGX;
  MonitorElement * DiMuonMassRC_LOG10;
  MonitorElement * DiMuonMassWC_LOG10;

  MonitorElement * DeltaEtaMuons;
  MonitorElement * DeltaPhiMuons;
  MonitorElement * MuonEfficiency_pT;
  MonitorElement * MuonEfficiency_eta;
  MonitorElement * TriggerEfficiencies;

};

#endif

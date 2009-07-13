#ifndef TopHLTDiMuonDQM_H
#define TopHLTDiMuonDQM_H

/*
 *  DQM HLT Dimuon Test Client
 *
 *  $Date: 2008/10/16 16:43:28 $
 *  $Revision: 1.2 $
 *  \author  M. Vander Donckt CERN
 *   
 */

#include <memory>
#include <unistd.h>
#include <stdlib.h>
#include <functional>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
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


//
// class declaration
//

class TopHLTDiMuonDQM : public edm::EDAnalyzer {

 public:

  TopHLTDiMuonDQM( const edm::ParameterSet& );
  ~TopHLTDiMuonDQM();

 protected:   

  void beginJob(const edm::EventSetup& c);
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
  std::string outputFile_;

  int level_;
  int counterEvt_;
  int prescaleEvt_;
  double coneSize_;

  edm::InputTag candCollectionTag_;

  // ----------member data ---------------------------

  bool verbose_;

  MonitorElement * NMuons;
  MonitorElement * PtMuons;
  MonitorElement * EtaMuons;
  MonitorElement * PhiMuons;
  MonitorElement * DiMuonMass;
  MonitorElement * DeltaEtaMuons;
  MonitorElement * DeltaPhiMuons;

};

#endif

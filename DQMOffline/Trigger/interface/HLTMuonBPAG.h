#ifndef DQMOffline_Trigger_HLTTopPlotter_H
#define DQMOffline_Trigger_HLTTopPlotter_H

/** \class HLTTopPlotter
 *  Get L1/HLT efficiency/rate plots
 *  Documentation available on the CMS TWiki:
 *  https://twiki.cern.ch/twiki/bin/view/CMS/MuonHLTOfflinePerformance
 *
 *  $Date: 2010/03/16 14:36:39 $
 *  $Revision: 1.5 $
 *  
 *  \author  J. Slaunwhite (modified from above
 */

// Base Class Headers
#include "DQMOffline/Trigger/interface/HLTMuonMatchAndPlot.h"
//#include "DQMOffline/Trigger/interface/MuonInformationDump.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
//#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/Math/interface/LorentzVector.h"

#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/MuonReco/interface/Muon.h"
//#include "CommonTools/Utilities/interface/StringCutObjectSelector.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include <vector>
#include "TFile.h"
#include "TString.h"
#include "TNtuple.h"



typedef math::XYZTLorentzVector LorentzVector;

class HLTMuonBPAG: public HLTMuonMatchAndPlot {

  // Inheritance means TopPlotter has all the same
  // members and functions as HLTMuonMatchAndPlot
  // In this header we can define extras for top

  
public:

  /// Constructor
  HLTMuonBPAG( const edm::ParameterSet& pset, std::string triggerName,
               std::vector<std::string> moduleNames,
               MuonSelectionStruct inputSelection,
               MuonSelectionStruct inputTagSelection,
               std::string customName,
               std::vector<std::string> validTriggers,
               const edm::Run & currentRun,
               const edm::EventSetup & currentEventSetup);

  // Operations
  void            begin  ( );
  void            analyze( const edm::Event & iEvent );
  void            finish ( );
  //MonitorElement* bookIt ( TString name, TString title, std::vector<double> );

  bool selectAndMatchMuons(const edm::Event & iEvent,
                           std::vector<MatchStruct> & myRecMatches,
                           std::vector< std::vector<HltFakeStruct> > & myHltFakeCands
                           );

  
private:

  TString ALLKEY;
  
  
  
  std::map <TString, MonitorElement*> diMuonMassVsPt;
  std::map <TString, MonitorElement*> diMuonMassVsEta;
  std::map <TString, MonitorElement*> diMuonMassVsPhi;
  std::map <TString, MonitorElement*> diMuonMass;
  std::map <TString, MonitorElement*> probeMuonPt;

  std::vector<double> theMassParameters;
  
  //void sortJets (reco::CaloJetCollection & theJets);

  MuonSelectionStruct tagSelection;
  std::vector<MatchStruct> tagRecMatches;
  std::vector< std::vector<HltFakeStruct> >  tagHltFakeCands;

  MonitorElement* book2DVarBins (TString name, TString title, int nBinsX, double * xBinLowEdges, int nBinsY, double yMin, double yMax);
  
};
#endif

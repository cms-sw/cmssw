#ifndef HLTriggerOffline_Muon_HLTMuonPlotter_H
#define HLTriggerOffline_Muon_HLTMuonPlotter_H

/** \class HLTMuonPlotter
 *  Generate histograms for muon trigger efficiencies
 *  Documentation available on the CMS TWiki:
 *  https://twiki.cern.ch/twiki/bin/view/CMS/MuonHLTOfflinePerformance
 *
 *  $Date: 2013/04/19 23:22:27 $
 *  $Revision: 1.3 $
 *  \author  J. Klukas, M. Vander Donckt, J. Alcaraz
 */

#include "HLTriggerOffline/Muon/interface/L1MuonMatcherAlgo.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"


#include <vector>
#include <set>
#include <map>
#include <iostream>
#include <algorithm>
#include <cctype>

#include "TPRegexp.h"



const unsigned int kNull = (unsigned int) -1;



class HLTMuonPlotter {

 public:
  HLTMuonPlotter(const edm::ParameterSet &, std::string,
                 const std::vector<std::string>&, const std::vector<std::string>&);
  void beginJob();
  void beginRun(const edm::Run &, const edm::EventSetup &);
  void analyze(const edm::Event &, const edm::EventSetup &);

 private:
  
  struct MatchStruct {
    const reco::Candidate            * candBase;
    const l1extra::L1MuonParticle    * candL1;
    std::vector<const reco::RecoChargedCandidate *> candHlt;
    MatchStruct() {
      candBase   = 0;
      candL1     = 0;
    }
    MatchStruct(const reco::Candidate * cand) {
      candBase = cand;
      candL1     = 0;
    }
    bool operator<(MatchStruct match) {
      return candBase->pt() < match.candBase->pt();
    }
    bool operator>(MatchStruct match) {
      return candBase->pt() > match.candBase->pt();
    }
  };
  struct matchesByDescendingPt {
    bool operator() (MatchStruct a, MatchStruct b) {
      return a.candBase->pt() > b.candBase->pt();
    }
  };

  void analyzePath(const edm::Event &, 
                   const std::string &, const std::string &,
                   const std::vector<MatchStruct>&, 
                   edm::Handle<trigger::TriggerEventWithRefs>);
  void findMatches(
      std::vector<MatchStruct> &, 
      const std::vector<l1extra::L1MuonParticleRef>&,
      const std::vector< std::vector< const reco::RecoChargedCandidate *> >&
      );
  void bookHist(std::string, std::string, std::string, std::string);

  std::string  hltPath_;
  std::string  hltProcessName_;

  std::vector<std::string> moduleLabels_;
  std::vector<std::string> stepLabels_;

  std::string genParticleLabel_;
  std::string     recMuonLabel_;

  std::vector<double> parametersEta_;
  std::vector<double> parametersPhi_;
  std::vector<double> parametersTurnOn_;

  double cutMinPt_;
  double cutMaxEta_;
  unsigned int cutMotherId_;
  std::vector<double> cutsDr_;
  std::string genMuonCut_;
  std::string recMuonCut_;

  StringCutObjectSelector<reco::GenParticle> * genMuonSelector_;
  StringCutObjectSelector<reco::Muon       > * recMuonSelector_;

  L1MuonMatcherAlgo l1Matcher_;

  DQMStore* dbe_;
  std::map<std::string, MonitorElement *> elements_;

};

#endif

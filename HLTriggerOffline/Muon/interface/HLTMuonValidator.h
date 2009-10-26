#ifndef HLTriggerOffline_Muon_HLTMuonValidator_H
#define HLTriggerOffline_Muon_HLTMuonValidator_H

/** \class HLTMuonValidator
 *  Generate histograms for muon trigger efficiencies
 *  Documentation available on the CMS TWiki:
 *  https://twiki.cern.ch/twiki/bin/view/CMS/MuonHLTOfflinePerformance
 *
 *  $Date: 2009/10/15 19:57:44 $
 *  $Revision: 1.1 $
 *  \author  J. Klukas, M. Vander Donckt, J. Alcaraz
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <vector>
#include <set>
#include <map>
#include <iostream>
#include <algorithm>
#include <cctype>

#include "TPRegexp.h"



const std::string kSources[] = {"gen", "rec"};
const unsigned int kNull = (unsigned int) -1;
TPRegexp kLooseL1Requirement("HLT_Mu3|Double");



class HLTMuonValidator : public edm::EDAnalyzer {

 public:
  HLTMuonValidator(const edm::ParameterSet &);
  virtual void beginJob();
  virtual void analyze(const edm::Event &, const edm::EventSetup &);

 private:
  
  struct MatchStruct {
    const reco::Candidate            * candBase;
    const l1extra::L1MuonParticle    * candL1;
    const reco::RecoChargedCandidate * candHlt[2];
    const reco::RecoChargedCandidate * candL2() {return candHlt[0];}
    const reco::RecoChargedCandidate * candL3() {return candHlt[1];}
    void setL1(const l1extra::L1MuonParticle    * cand) {candL1     = cand;}
    void setL2(const reco::RecoChargedCandidate * cand) {candHlt[0] = cand;}
    void setL3(const reco::RecoChargedCandidate * cand) {candHlt[1] = cand;}
    MatchStruct() {
      candBase   = 0;
      candL1     = 0;
      candHlt[0] = 0;
      candHlt[1] = 0;
    }
    MatchStruct(const reco::Candidate * cand) {
      candBase = cand;
      candL1     = 0;
      candHlt[0] = 0;
      candHlt[1] = 0;
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

  void analyzePath(const std::string &, const std::string &,
                   const std::vector<MatchStruct> &, 
                   const trigger::TriggerEventWithRefs &);
  const reco::Candidate * findMother(const reco::Candidate *);
  bool identical(const reco::Candidate *, const reco::Candidate *);
  unsigned int findMatch(const reco::Candidate *, std::vector<MatchStruct> &, 
                         double, std::string);
  void bookHist(std::string, std::string, std::string, std::string);

  std::string  hltProcessName_;

  std::vector<std::string> hltPathsToCheck_;
  std::set<std::string> hltPaths_;
  std::map< std::string, std::vector<std::string> > filterLabels_;

  std::vector<double> parametersEta_;
  std::vector<double> parametersPhi_;
  std::vector<double> parametersTurnOn_;

  double       cutMinPt_;
  double       cutMaxEta_;
  unsigned int cutMotherId_;
  std::vector<double> cutsDr_;

  DQMStore* dbe_;
  std::map<std::string, MonitorElement *> elements_;
  std::map<std::string, std::vector<std::string> > stepLabels_;

};

#endif

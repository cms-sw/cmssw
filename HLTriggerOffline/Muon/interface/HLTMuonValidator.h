#ifndef HLTriggerOffline_Muon_HLTMuonValidator_H
#define HLTriggerOffline_Muon_HLTMuonValidator_H

/** \class HLTMuonValidator
 *  Generate histograms for muon trigger efficiencies
 *  Documentation available on the CMS TWiki:
 *  https://twiki.cern.ch/twiki/bin/view/CMS/MuonHLTOfflinePerformance
 *
 *  $Date: 2013/04/19 23:22:27 $
 *  $Revision: 1.13 $
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

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

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



class HLTMuonValidator : public edm::EDAnalyzer {

 public:
  HLTMuonValidator(const edm::ParameterSet &);
  virtual void beginJob();
  virtual void beginRun(const edm::Run &, const edm::EventSetup &);
  virtual void analyze(const edm::Event &, const edm::EventSetup &);

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

  void initializeHists();
  void analyzePath(const edm::Event &, 
                   const std::string &, const std::string &,
                   const std::vector<MatchStruct>&, 
                   edm::Handle<trigger::TriggerEventWithRefs>);
  void findMatches(
      std::vector<MatchStruct> &, 
      std::vector<l1extra::L1MuonParticleRef>,
      std::vector< std::vector< const reco::RecoChargedCandidate *> >
      );
  void bookHist(std::string, std::string, std::string, std::string);

  std::string  hltProcessName_;

  std::vector<std::string> hltPathsToCheck_;
  std::set   <std::string> hltPaths_;
  std::map   <std::string, std::vector<std::string> > filterLabels_;

  std::string genParticleLabel_;
  std::string     recMuonLabel_;
  std::string      l1CandLabel_;
  std::string      l2CandLabel_;
  std::string      l3CandLabel_;

  std::vector<double> parametersEta_;
  std::vector<double> parametersPhi_;
  std::vector<double> parametersTurnOn_;

  std::map<std::string, double> cutsMinPt_;

  double       cutMaxEta_;
  unsigned int cutMotherId_;
  std::vector<double> cutsDr_;
  std::string genMuonCut_;
  std::string recMuonCut_;

  StringCutObjectSelector<reco::GenParticle> * genMuonSelector_;
  StringCutObjectSelector<reco::Muon       > * recMuonSelector_;

  HLTConfigProvider hltConfig_;

  L1MuonMatcherAlgo l1Matcher_;

  DQMStore* dbe_;
  std::map<std::string, MonitorElement *> elements_;
  std::map<std::string, std::vector<std::string> > stepLabels_;

};

#endif

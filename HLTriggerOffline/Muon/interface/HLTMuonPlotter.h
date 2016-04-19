#ifndef HLTriggerOffline_Muon_HLTMuonPlotter_H
#define HLTriggerOffline_Muon_HLTMuonPlotter_H

/** \class HLTMuonPlotter
 *  Generate histograms for muon trigger efficiencies
 *  Documentation available on the CMS TWiki:
 *  https://twiki.cern.ch/twiki/bin/view/CMS/MuonHLTOfflinePerformance
 *
 *  \author  J. Klukas, M. Vander Donckt, J. Alcaraz
 */

#include "HLTriggerOffline/Muon/interface/L1MuonMatcherAlgo.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/L1Trigger/interface/Muon.h"

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
#include "boost/tuple/tuple.hpp"

#include "TPRegexp.h"



const unsigned int kNull = (unsigned int) -1;



class HLTMuonPlotter {

 public:
  HLTMuonPlotter(const edm::ParameterSet &, std::string,
                 const std::vector<std::string>&, 
		 const std::vector<std::string>&,
		 const boost::tuple<edm::EDGetTokenT<trigger::TriggerEventWithRefs>,
		                    edm::EDGetTokenT<reco::GenParticleCollection>,
		                    edm::EDGetTokenT<reco::MuonCollection> >&
		 );

  ~HLTMuonPlotter() {
    delete genMuonSelector_;
    delete recMuonSelector_;
  }

  void beginJob();
  void beginRun(DQMStore::IBooker &, const edm::Run &, const edm::EventSetup &);
  void analyze(const edm::Event &, const edm::EventSetup &);

  static boost::tuple<
    edm::EDGetTokenT<trigger::TriggerEventWithRefs>,
    edm::EDGetTokenT<reco::GenParticleCollection>,
    edm::EDGetTokenT<reco::MuonCollection> > getTokens(const edm::ParameterSet&, edm::ConsumesCollector&&);
  
 private:
  
  struct MatchStruct {
    const reco::Candidate            * candBase;
    const l1t::Muon                  * candL1;
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
      const l1t::MuonVectorRef& candsL1,
      const std::vector< std::vector< const reco::RecoChargedCandidate *> >&
      );
  void bookHist(DQMStore::IBooker &, std::string,
		std::string, std::string, std::string);

  std::string  hltPath_;
  std::string  hltProcessName_;

  std::vector<std::string> moduleLabels_;
  std::vector<std::string> stepLabels_;

  edm::EDGetTokenT<trigger::TriggerEventWithRefs> hltTriggerSummaryRAW_;
  edm::EDGetTokenT<reco::GenParticleCollection> genParticleLabel_;
  edm::EDGetTokenT<reco::MuonCollection> recMuonLabel_;

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

  std::map<std::string, MonitorElement *> elements_;

};

#endif

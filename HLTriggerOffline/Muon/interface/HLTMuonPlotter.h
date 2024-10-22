#ifndef HLTriggerOffline_Muon_HLTMuonPlotter_h
#define HLTriggerOffline_Muon_HLTMuonPlotter_h

/** \class HLTMuonPlotter
 *  Generate histograms for muon trigger efficiencies
 *  Documentation available on the CMS TWiki:
 *  https://twiki.cern.ch/twiki/bin/view/CMS/MuonHLTOfflinePerformance
 *
 *  \author  J. Klukas, M. Vander Donckt, J. Alcaraz
 */

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "MuonAnalysis/MuonAssociators/interface/L1MuonMatcherAlgo.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>

#include "TPRegexp.h"

class HLTMuonPlotter {
public:
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;
  typedef L1MuonMatcherAlgoT<edm::Transition::BeginRun> L1MuonMatcherAlgoForDQM;

  HLTMuonPlotter(const edm::ParameterSet &,
                 const std::string &,
                 const std::vector<std::string> &,
                 const std::vector<std::string> &,
                 const edm::EDGetTokenT<trigger::TriggerEventWithRefs> &,
                 const edm::EDGetTokenT<reco::GenParticleCollection> &,
                 const edm::EDGetTokenT<reco::MuonCollection> &,
                 const L1MuonMatcherAlgoForDQM &);

  ~HLTMuonPlotter() = default;

  void beginRun(DQMStore::IBooker &, const edm::Run &, const edm::EventSetup &);
  void analyze(const edm::Event &, const edm::EventSetup &);

private:
  struct MatchStruct {
    const reco::Candidate *candBase;
    const l1t::Muon *candL1;
    std::vector<const reco::RecoChargedCandidate *> candHlt;
    MatchStruct() {
      candBase = nullptr;
      candL1 = nullptr;
    }
    MatchStruct(const reco::Candidate *cand) {
      candBase = cand;
      candL1 = nullptr;
    }
    bool operator<(MatchStruct match) { return candBase->pt() < match.candBase->pt(); }
    bool operator>(MatchStruct match) { return candBase->pt() > match.candBase->pt(); }
  };
  struct matchesByDescendingPt {
    bool operator()(MatchStruct a, MatchStruct b) { return a.candBase->pt() > b.candBase->pt(); }
  };

  void findMatches(std::vector<MatchStruct> &,
                   const l1t::MuonVectorRef &candsL1,
                   const std::vector<std::vector<const reco::RecoChargedCandidate *>> &);

  void bookHist(DQMStore::IBooker &, const std::string &, const std::string &, const std::string &, const std::string &);

  template <typename T>
  std::string vector_to_string(std::vector<T> const &vec, std::string const &delimiter = " ") const;

  std::string const hltPath_;
  std::string const hltProcessName_;

  std::vector<std::string> const moduleLabels_;
  std::vector<std::string> const stepLabels_;

  edm::EDGetTokenT<trigger::TriggerEventWithRefs> const triggerEventWithRefsToken_;
  edm::EDGetTokenT<reco::GenParticleCollection> const genParticleToken_;
  edm::EDGetTokenT<reco::MuonCollection> const recMuonToken_;

  StringCutObjectSelector<reco::GenParticle> const genMuonSelector_;
  StringCutObjectSelector<reco::Muon> const recMuonSelector_;
  std::vector<double> const cutsDr_;

  std::vector<double> const parametersEta_;
  std::vector<double> const parametersPhi_;
  std::vector<double> const parametersTurnOn_;

  L1MuonMatcherAlgoForDQM l1Matcher_;

  bool isInvalid_;

  double cutMinPt_;
  double cutMaxEta_;

  std::unordered_map<std::string, MonitorElement *> elements_;
};

template <typename T>
std::string HLTMuonPlotter::vector_to_string(std::vector<T> const &vec, std::string const &delimiter) const {
  if (vec.empty())
    return "";
  std::stringstream sstr;
  for (auto const &foo : vec)
    sstr << delimiter << foo;
  auto ret = sstr.str();
  ret.erase(0, delimiter.size());
  return ret;
}

#endif  // HLTriggerOffline_Muon_HLTMuonPlotter_h

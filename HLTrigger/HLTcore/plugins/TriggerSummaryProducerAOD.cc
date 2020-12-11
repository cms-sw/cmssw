/** \class TriggerSummaryProducerAOD
 *
 * See header file for documentation
 *
 *
 *  \author Martin Grunewald
 *
 */

#include <ostream>
#include <algorithm>
#include <memory>
#include <typeinfo>

#include "HLTrigger/HLTcore/interface/TriggerSummaryProducerAOD.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/Framework/interface/ProcessMatch.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETFwd.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETFwd.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"

#include "DataFormats/L1Trigger/interface/L1HFRings.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"

#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

#include "DataFormats/L1TCorrelator/interface/TkMuon.h"
#include "DataFormats/L1TCorrelator/interface/TkElectron.h"
#include "DataFormats/L1TCorrelator/interface/TkEm.h"
#include "DataFormats/L1TParticleFlow/interface/PFJet.h"
#include "DataFormats/L1TParticleFlow/interface/PFTau.h"
#include "DataFormats/L1TParticleFlow/interface/HPSPFTau.h"
#include "DataFormats/L1TParticleFlow/interface/HPSPFTauFwd.h"
#include "DataFormats/L1TParticleFlow/interface/PFTrack.h"

#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/TauReco/interface/PFTau.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"

#include "boost/algorithm/string.hpp"

namespace {
  std::vector<std::regex> convertToRegex(std::vector<std::string> const& iPatterns) {
    std::vector<std::regex> result;

    for (auto const& pattern : iPatterns) {
      auto regexPattern = pattern;
      boost::replace_all(regexPattern, "*", ".*");
      boost::replace_all(regexPattern, "?", ".");

      result.emplace_back(regexPattern);
    }
    return result;
  }
}  // namespace

//
// constructors and destructor
//
TriggerSummaryProducerAOD::TriggerSummaryProducerAOD(const edm::ParameterSet& ps)
    : throw_(ps.getParameter<bool>("throw")),
      pn_(ps.getParameter<std::string>("processName")),
      moduleLabelPatternsToMatch_(
          convertToRegex(ps.getParameter<std::vector<std::string>>("moduleLabelPatternsToMatch"))),
      moduleLabelPatternsToSkip_(
          convertToRegex(ps.getParameter<std::vector<std::string>>("moduleLabelPatternsToSkip"))) {
  if (pn_ == "@") {
    edm::Service<edm::service::TriggerNamesService> tns;
    if (tns.isAvailable()) {
      pn_ = tns->getProcessName();
    } else {
      edm::LogError("TriggerSummaryProducerAOD") << "HLT Error: TriggerNamesService not available!";
      pn_ = "*";
    }
  }
  LogDebug("TriggerSummaryProducerAOD") << "Using process name: '" << pn_ << "'";

  produces<trigger::TriggerEvent>();

  auto const* pProcessName = &pn_;
  auto const& moduleLabelPatternsToMatch = moduleLabelPatternsToMatch_;
  auto const& moduleLabelPatternsToSkip = moduleLabelPatternsToSkip_;
  auto productMatch = [pProcessName, &moduleLabelPatternsToSkip, &moduleLabelPatternsToMatch](
                          edm::BranchDescription const& iBranch) -> bool {
    if (iBranch.processName() == *pProcessName || *pProcessName == "*") {
      auto const& label = iBranch.moduleLabel();
      for (auto& match : moduleLabelPatternsToMatch) {
        if (std::regex_match(label, match)) {
          //make sure this is not in the reject list
          for (auto& reject : moduleLabelPatternsToSkip) {
            if (std::regex_match(label, reject)) {
              return false;
            }
          }
          return true;
        }
      }
    }
    return false;
  };

  getTriggerFilterObjectWithRefs_ = edm::GetterOfProducts<trigger::TriggerFilterObjectWithRefs>(productMatch, this);
  getRecoEcalCandidateCollection_ = edm::GetterOfProducts<reco::RecoEcalCandidateCollection>(productMatch, this);
  getElectronCollection_ = edm::GetterOfProducts<reco::ElectronCollection>(productMatch, this);
  getRecoChargedCandidateCollection_ = edm::GetterOfProducts<reco::RecoChargedCandidateCollection>(productMatch, this);
  getCaloJetCollection_ = edm::GetterOfProducts<reco::CaloJetCollection>(productMatch, this);
  getCompositeCandidateCollection_ = edm::GetterOfProducts<reco::CompositeCandidateCollection>(productMatch, this);
  getMETCollection_ = edm::GetterOfProducts<reco::METCollection>(productMatch, this);
  getCaloMETCollection_ = edm::GetterOfProducts<reco::CaloMETCollection>(productMatch, this);
  getIsolatedPixelTrackCandidateCollection_ =
      edm::GetterOfProducts<reco::IsolatedPixelTrackCandidateCollection>(productMatch, this);
  getL1EmParticleCollection_ = edm::GetterOfProducts<l1extra::L1EmParticleCollection>(productMatch, this);
  getL1MuonParticleCollection_ = edm::GetterOfProducts<l1extra::L1MuonParticleCollection>(productMatch, this);
  getL1JetParticleCollection_ = edm::GetterOfProducts<l1extra::L1JetParticleCollection>(productMatch, this);
  getL1EtMissParticleCollection_ = edm::GetterOfProducts<l1extra::L1EtMissParticleCollection>(productMatch, this);
  getL1HFRingsCollection_ = edm::GetterOfProducts<l1extra::L1HFRingsCollection>(productMatch, this);
  getL1TMuonParticleCollection_ = edm::GetterOfProducts<l1t::MuonBxCollection>(productMatch, this);
  getL1TEGammaParticleCollection_ = edm::GetterOfProducts<l1t::EGammaBxCollection>(productMatch, this);
  getL1TJetParticleCollection_ = edm::GetterOfProducts<l1t::JetBxCollection>(productMatch, this);
  getL1TTauParticleCollection_ = edm::GetterOfProducts<l1t::TauBxCollection>(productMatch, this);
  getL1TEtSumParticleCollection_ = edm::GetterOfProducts<l1t::EtSumBxCollection>(productMatch, this);

  getL1TTkMuonCollection_ = edm::GetterOfProducts<l1t::TkMuonCollection>(productMatch, this);
  getL1TTkElectronCollection_ = edm::GetterOfProducts<l1t::TkElectronCollection>(productMatch, this);
  getL1TTkEmCollection_ = edm::GetterOfProducts<l1t::TkEmCollection>(productMatch, this);
  getL1TPFJetCollection_ = edm::GetterOfProducts<l1t::PFJetCollection>(productMatch, this);
  getL1TPFTauCollection_ = edm::GetterOfProducts<l1t::PFTauCollection>(productMatch, this);
  getL1THPSPFTauCollection_ = edm::GetterOfProducts<l1t::HPSPFTauCollection>(productMatch, this);
  getL1TPFTrackCollection_ = edm::GetterOfProducts<l1t::PFTrackCollection>(productMatch, this);

  getPFJetCollection_ = edm::GetterOfProducts<reco::PFJetCollection>(productMatch, this);
  getPFTauCollection_ = edm::GetterOfProducts<reco::PFTauCollection>(productMatch, this);
  getPFMETCollection_ = edm::GetterOfProducts<reco::PFMETCollection>(productMatch, this);

  callWhenNewProductsRegistered([this](edm::BranchDescription const& bd) {
    getTriggerFilterObjectWithRefs_(bd);
    getRecoEcalCandidateCollection_(bd);
    getElectronCollection_(bd);
    getRecoChargedCandidateCollection_(bd);
    getCaloJetCollection_(bd);
    getCompositeCandidateCollection_(bd);
    getMETCollection_(bd);
    getCaloMETCollection_(bd);
    getIsolatedPixelTrackCandidateCollection_(bd);
    getL1EmParticleCollection_(bd);
    getL1MuonParticleCollection_(bd);
    getL1JetParticleCollection_(bd);
    getL1EtMissParticleCollection_(bd);
    getL1HFRingsCollection_(bd);
    getL1TMuonParticleCollection_(bd);
    getL1TEGammaParticleCollection_(bd);
    getL1TJetParticleCollection_(bd);
    getL1TTauParticleCollection_(bd);
    getL1TEtSumParticleCollection_(bd);
    getL1TTkMuonCollection_(bd);
    getL1TTkElectronCollection_(bd);
    getL1TTkEmCollection_(bd);
    getL1TPFJetCollection_(bd);
    getL1TPFTauCollection_(bd);
    getL1THPSPFTauCollection_(bd);
    getL1TPFTrackCollection_(bd);
    getPFJetCollection_(bd);
    getPFTauCollection_(bd);
    getPFMETCollection_(bd);
  });
}

TriggerSummaryProducerAOD::~TriggerSummaryProducerAOD() = default;

//
// member functions
//

namespace {
  inline void tokenizeTag(const std::string& tag, std::string& label, std::string& instance, std::string& process) {
    using std::string;

    const char token(':');
    const string empty;

    label = tag;
    const string::size_type i1(label.find(token));
    if (i1 == string::npos) {
      instance = empty;
      process = empty;
    } else {
      instance = label.substr(i1 + 1);
      label.resize(i1);
      const string::size_type i2(instance.find(token));
      if (i2 == string::npos) {
        process = empty;
      } else {
        process = instance.substr(i2 + 1);
        instance.resize(i2);
      }
    }
  }
}  // namespace

void TriggerSummaryProducerAOD::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("throw", false)->setComment("Throw exception or LogError");
  desc.add<std::string>("processName", "@")
      ->setComment(
          "Process name to use when getting data. The value of '@' is used to denote the current process name.");
  desc.add<std::vector<std::string>>("moduleLabelPatternsToMatch", std::vector<std::string>(1, "hlt*"))
      ->setComment("glob patterns for module labels to get data.");
  desc.add<std::vector<std::string>>("moduleLabelPatternsToSkip", std::vector<std::string>())
      ->setComment("module labels for data products which should not be gotten.");
  descriptions.add("triggerSummaryProducerAOD", desc);
}

// ------------ method called to produce the data  ------------
void TriggerSummaryProducerAOD::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace l1extra;
  using namespace trigger;
  using namespace l1t;

  std::vector<edm::Handle<trigger::TriggerFilterObjectWithRefs>> fobs;
  getTriggerFilterObjectWithRefs_.fillHandles(iEvent, fobs);

  const unsigned int nfob(fobs.size());
  LogTrace("TriggerSummaryProducerAOD") << "Number of filter  objects found: " << nfob;

  string tagLabel, tagInstance, tagProcess;

  ///
  /// check whether collection tags are recorded in filterobjects; if
  /// so, these are L3 collections to be packed up, and the
  /// corresponding filter is a L3 filter also to be packed up.
  /// Record the InputTags of those L3 filters and L3 collections
  std::vector<bool> maskFilters;
  maskFilters.resize(nfob);
  InputTagSet filterTagsEvent(pn_ != "*");
  InputTagSet collectionTagsEvent(pn_ != "*");

  unsigned int nf(0);
  for (unsigned int ifob = 0; ifob != nfob; ++ifob) {
    maskFilters[ifob] = false;
    const vector<string>& collectionTags_(fobs[ifob]->getCollectionTagsAsStrings());
    const unsigned int ncol(collectionTags_.size());
    if (ncol > 0) {
      nf++;
      maskFilters[ifob] = true;
      const string& label(fobs[ifob].provenance()->moduleLabel());
      const string& instance(fobs[ifob].provenance()->productInstanceName());
      const string& process(fobs[ifob].provenance()->processName());
      filterTagsEvent.insert(InputTag(label, instance, process));
      for (unsigned int icol = 0; icol != ncol; ++icol) {
        // overwrite process name (usually not set)
        tokenizeTag(collectionTags_[icol], tagLabel, tagInstance, tagProcess);
        collectionTagsEvent.insert(InputTag(tagLabel, tagInstance, pn_));
      }
    }
  }
  /// check uniqueness count
  if (filterTagsEvent.size() != nf) {
    LogError("TriggerSummaryProducerAOD")
        << "Mismatch in number of filter tags: " << filterTagsEvent.size() << "!=" << nf;
  }

  /// accumulate for endJob printout
  collectionTagsGlobal_.insert(collectionTagsEvent.begin(), collectionTagsEvent.end());
  filterTagsGlobal_.insert(filterTagsEvent.begin(), filterTagsEvent.end());

  /// debug printout
  if (isDebugEnabled()) {
    /// event-by-event tags
    const unsigned int nc(collectionTagsEvent.size());
    LogTrace("TriggerSummaryProducerAOD") << "Number of unique collections requested " << nc;
    const InputTagSet::const_iterator cb(collectionTagsEvent.begin());
    const InputTagSet::const_iterator ce(collectionTagsEvent.end());
    for (InputTagSet::const_iterator ci = cb; ci != ce; ++ci) {
      LogTrace("TriggerSummaryProducerAOD") << distance(cb, ci) << " " << ci->encode();
    }
    const unsigned int nf(filterTagsEvent.size());
    LogTrace("TriggerSummaryProducerAOD") << "Number of unique filters requested " << nf;
    const InputTagSet::const_iterator fb(filterTagsEvent.begin());
    const InputTagSet::const_iterator fe(filterTagsEvent.end());
    for (InputTagSet::const_iterator fi = fb; fi != fe; ++fi) {
      LogTrace("TriggerSummaryProducerAOD") << distance(fb, fi) << " " << fi->encode();
    }
  }

  ///
  /// Now the processing:
  /// first trigger objects from L3 collections, then L3 filter objects
  ///
  /// create trigger objects, fill triggerobjectcollection and offset map
  trigger::TriggerObjectCollection toc;
  //toc_.clear();
  std::vector<std::string> tags;
  trigger::Keys keys;
  std::map<edm::ProductID, unsigned int> offset;

  fillTriggerObjectCollections<RecoEcalCandidateCollection>(
      toc, offset, tags, keys, iEvent, getRecoEcalCandidateCollection_, collectionTagsEvent);
  fillTriggerObjectCollections<ElectronCollection>(
      toc, offset, tags, keys, iEvent, getElectronCollection_, collectionTagsEvent);
  fillTriggerObjectCollections<RecoChargedCandidateCollection>(
      toc, offset, tags, keys, iEvent, getRecoChargedCandidateCollection_, collectionTagsEvent);
  fillTriggerObjectCollections<CaloJetCollection>(
      toc, offset, tags, keys, iEvent, getCaloJetCollection_, collectionTagsEvent);
  fillTriggerObjectCollections<CompositeCandidateCollection>(
      toc, offset, tags, keys, iEvent, getCompositeCandidateCollection_, collectionTagsEvent);
  fillTriggerObjectCollections<METCollection>(toc, offset, tags, keys, iEvent, getMETCollection_, collectionTagsEvent);
  fillTriggerObjectCollections<CaloMETCollection>(
      toc, offset, tags, keys, iEvent, getCaloMETCollection_, collectionTagsEvent);
  fillTriggerObjectCollections<IsolatedPixelTrackCandidateCollection>(
      toc, offset, tags, keys, iEvent, getIsolatedPixelTrackCandidateCollection_, collectionTagsEvent);
  ///
  fillTriggerObjectCollections<L1EmParticleCollection>(
      toc, offset, tags, keys, iEvent, getL1EmParticleCollection_, collectionTagsEvent);
  fillTriggerObjectCollections<L1MuonParticleCollection>(
      toc, offset, tags, keys, iEvent, getL1MuonParticleCollection_, collectionTagsEvent);
  fillTriggerObjectCollections<L1JetParticleCollection>(
      toc, offset, tags, keys, iEvent, getL1JetParticleCollection_, collectionTagsEvent);
  fillTriggerObjectCollections<L1EtMissParticleCollection>(
      toc, offset, tags, keys, iEvent, getL1EtMissParticleCollection_, collectionTagsEvent);
  fillTriggerObjectCollections<L1HFRingsCollection>(
      toc, offset, tags, keys, iEvent, getL1HFRingsCollection_, collectionTagsEvent);
  fillTriggerObjectCollections<MuonBxCollection>(
      toc, offset, tags, keys, iEvent, getL1TMuonParticleCollection_, collectionTagsEvent);
  fillTriggerObjectCollections<EGammaBxCollection>(
      toc, offset, tags, keys, iEvent, getL1TEGammaParticleCollection_, collectionTagsEvent);
  fillTriggerObjectCollections<JetBxCollection>(
      toc, offset, tags, keys, iEvent, getL1TJetParticleCollection_, collectionTagsEvent);
  fillTriggerObjectCollections<TauBxCollection>(
      toc, offset, tags, keys, iEvent, getL1TTauParticleCollection_, collectionTagsEvent);
  fillTriggerObjectCollections<EtSumBxCollection>(
      toc, offset, tags, keys, iEvent, getL1TEtSumParticleCollection_, collectionTagsEvent);
  ///
  fillTriggerObjectCollections<l1t::TkMuonCollection>(
      toc, offset, tags, keys, iEvent, getL1TTkMuonCollection_, collectionTagsEvent);
  fillTriggerObjectCollections<l1t::TkElectronCollection>(
      toc, offset, tags, keys, iEvent, getL1TTkElectronCollection_, collectionTagsEvent);
  fillTriggerObjectCollections<l1t::TkEmCollection>(
      toc, offset, tags, keys, iEvent, getL1TTkEmCollection_, collectionTagsEvent);
  fillTriggerObjectCollections<l1t::PFJetCollection>(
      toc, offset, tags, keys, iEvent, getL1TPFJetCollection_, collectionTagsEvent);
  fillTriggerObjectCollections<l1t::PFTauCollection>(
      toc, offset, tags, keys, iEvent, getL1TPFTauCollection_, collectionTagsEvent);
  fillTriggerObjectCollections<l1t::HPSPFTauCollection>(
      toc, offset, tags, keys, iEvent, getL1THPSPFTauCollection_, collectionTagsEvent);
  fillTriggerObjectCollections<l1t::PFTrackCollection>(
      toc, offset, tags, keys, iEvent, getL1TPFTrackCollection_, collectionTagsEvent);
  ///
  fillTriggerObjectCollections<reco::PFJetCollection>(
      toc, offset, tags, keys, iEvent, getPFJetCollection_, collectionTagsEvent);
  fillTriggerObjectCollections<reco::PFTauCollection>(
      toc, offset, tags, keys, iEvent, getPFTauCollection_, collectionTagsEvent);
  fillTriggerObjectCollections<reco::PFMETCollection>(
      toc, offset, tags, keys, iEvent, getPFMETCollection_, collectionTagsEvent);
  ///
  const unsigned int nk(tags.size());
  LogDebug("TriggerSummaryProducerAOD") << "Number of collections found: " << nk;
  const unsigned int no(toc.size());
  LogDebug("TriggerSummaryProducerAOD") << "Number of physics objects found: " << no;

  ///
  /// construct single AOD product, reserving capacity
  unique_ptr<TriggerEvent> product(new TriggerEvent(pn_, nk, no, nf));

  /// fill trigger object collection
  product->addCollections(tags, keys);
  product->addObjects(toc);

  /// fill the L3 filter objects
  trigger::Vids ids;
  for (unsigned int ifob = 0; ifob != nfob; ++ifob) {
    if (maskFilters[ifob]) {
      const string& label(fobs[ifob].provenance()->moduleLabel());
      const string& instance(fobs[ifob].provenance()->productInstanceName());
      const string& process(fobs[ifob].provenance()->processName());
      const edm::InputTag filterTag(label, instance, process);
      ids.clear();
      keys.clear();
      fillFilterObjectMembers(iEvent, filterTag, fobs[ifob]->photonIds(), fobs[ifob]->photonRefs(), offset, keys, ids);
      fillFilterObjectMembers(
          iEvent, filterTag, fobs[ifob]->electronIds(), fobs[ifob]->electronRefs(), offset, keys, ids);
      fillFilterObjectMembers(iEvent, filterTag, fobs[ifob]->muonIds(), fobs[ifob]->muonRefs(), offset, keys, ids);
      fillFilterObjectMembers(iEvent, filterTag, fobs[ifob]->jetIds(), fobs[ifob]->jetRefs(), offset, keys, ids);
      fillFilterObjectMembers(
          iEvent, filterTag, fobs[ifob]->compositeIds(), fobs[ifob]->compositeRefs(), offset, keys, ids);
      fillFilterObjectMembers(
          iEvent, filterTag, fobs[ifob]->basemetIds(), fobs[ifob]->basemetRefs(), offset, keys, ids);
      fillFilterObjectMembers(
          iEvent, filterTag, fobs[ifob]->calometIds(), fobs[ifob]->calometRefs(), offset, keys, ids);
      fillFilterObjectMembers(
          iEvent, filterTag, fobs[ifob]->pixtrackIds(), fobs[ifob]->pixtrackRefs(), offset, keys, ids);
      fillFilterObjectMembers(iEvent, filterTag, fobs[ifob]->l1emIds(), fobs[ifob]->l1emRefs(), offset, keys, ids);
      fillFilterObjectMembers(iEvent, filterTag, fobs[ifob]->l1muonIds(), fobs[ifob]->l1muonRefs(), offset, keys, ids);
      fillFilterObjectMembers(iEvent, filterTag, fobs[ifob]->l1jetIds(), fobs[ifob]->l1jetRefs(), offset, keys, ids);
      fillFilterObjectMembers(
          iEvent, filterTag, fobs[ifob]->l1etmissIds(), fobs[ifob]->l1etmissRefs(), offset, keys, ids);
      fillFilterObjectMembers(
          iEvent, filterTag, fobs[ifob]->l1hfringsIds(), fobs[ifob]->l1hfringsRefs(), offset, keys, ids);
      fillFilterObjectMembers(
          iEvent, filterTag, fobs[ifob]->l1tmuonIds(), fobs[ifob]->l1tmuonRefs(), offset, keys, ids);
      fillFilterObjectMembers(
          iEvent, filterTag, fobs[ifob]->l1tegammaIds(), fobs[ifob]->l1tegammaRefs(), offset, keys, ids);
      fillFilterObjectMembers(iEvent, filterTag, fobs[ifob]->l1tjetIds(), fobs[ifob]->l1tjetRefs(), offset, keys, ids);
      fillFilterObjectMembers(iEvent, filterTag, fobs[ifob]->l1ttauIds(), fobs[ifob]->l1ttauRefs(), offset, keys, ids);
      fillFilterObjectMembers(
          iEvent, filterTag, fobs[ifob]->l1tetsumIds(), fobs[ifob]->l1tetsumRefs(), offset, keys, ids);
      /**/
      fillFilterObjectMembers(
          iEvent, filterTag, fobs[ifob]->l1ttkmuonIds(), fobs[ifob]->l1ttkmuonRefs(), offset, keys, ids);
      fillFilterObjectMembers(
          iEvent, filterTag, fobs[ifob]->l1ttkeleIds(), fobs[ifob]->l1ttkeleRefs(), offset, keys, ids);
      fillFilterObjectMembers(
          iEvent, filterTag, fobs[ifob]->l1ttkemIds(), fobs[ifob]->l1ttkemRefs(), offset, keys, ids);
      fillFilterObjectMembers(
          iEvent, filterTag, fobs[ifob]->l1tpfjetIds(), fobs[ifob]->l1tpfjetRefs(), offset, keys, ids);
      fillFilterObjectMembers(
          iEvent, filterTag, fobs[ifob]->l1tpftauIds(), fobs[ifob]->l1tpftauRefs(), offset, keys, ids);
      fillFilterObjectMembers(
          iEvent, filterTag, fobs[ifob]->l1thpspftauIds(), fobs[ifob]->l1thpspftauRefs(), offset, keys, ids);
      fillFilterObjectMembers(
          iEvent, filterTag, fobs[ifob]->l1tpftrackIds(), fobs[ifob]->l1tpftrackRefs(), offset, keys, ids);
      /**/
      fillFilterObjectMembers(iEvent, filterTag, fobs[ifob]->pfjetIds(), fobs[ifob]->pfjetRefs(), offset, keys, ids);
      fillFilterObjectMembers(iEvent, filterTag, fobs[ifob]->pftauIds(), fobs[ifob]->pftauRefs(), offset, keys, ids);
      fillFilterObjectMembers(iEvent, filterTag, fobs[ifob]->pfmetIds(), fobs[ifob]->pfmetRefs(), offset, keys, ids);
      product->addFilter(filterTag, ids, keys);
    }
  }

  OrphanHandle<TriggerEvent> ref = iEvent.put(std::move(product));
  LogTrace("TriggerSummaryProducerAOD") << "Number of physics objects packed: " << ref->sizeObjects();
  LogTrace("TriggerSummaryProducerAOD") << "Number of filter  objects packed: " << ref->sizeFilters();
}

template <typename C>
void TriggerSummaryProducerAOD::fillTriggerObjectCollections(trigger::TriggerObjectCollection& toc,
                                                             ProductIDtoIndex& offset,
                                                             std::vector<std::string>& tags,
                                                             trigger::Keys& keys,
                                                             const edm::Event& iEvent,
                                                             const edm::GetterOfProducts<C>& getter,
                                                             const InputTagSet& collectionTagsEvent) const {
  /// this routine accesses the original (L3) collections (with C++
  /// typename C), extracts 4-momentum and id of each collection
  /// member, and packs this up

  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace l1extra;
  using namespace trigger;
  using namespace l1t;

  vector<Handle<C>> collections;
  getter.fillHandles(iEvent, collections);
  const unsigned int nc(collections.size());

  for (unsigned int ic = 0; ic != nc; ++ic) {
    const Provenance& provenance(*(collections[ic].provenance()));
    const string& label(provenance.moduleLabel());
    const string& instance(provenance.productInstanceName());
    const string& process(provenance.processName());
    const InputTag collectionTag(label, instance, process);

    if (collectionTagsEvent.find(collectionTag) != collectionTagsEvent.end()) {
      const ProductID pid(collections[ic].provenance()->productID());
      if (offset.find(pid) != offset.end()) {
        LogError("TriggerSummaryProducerAOD") << "Duplicate pid: " << pid;
      }
      offset[pid] = toc.size();
      const unsigned int n(collections[ic]->size());
      for (unsigned int i = 0; i != n; ++i) {
        fillTriggerObject(toc, (*collections[ic])[i]);
      }
      tags.push_back(collectionTag.encode());
      keys.push_back(toc.size());
    }

  }  /// end loop over handles
}

template <typename T>
void TriggerSummaryProducerAOD::fillTriggerObject(trigger::TriggerObjectCollection& toc, const T& object) const {
  using namespace trigger;
  toc.emplace_back(object);

  return;
}

void TriggerSummaryProducerAOD::fillTriggerObject(trigger::TriggerObjectCollection& toc,
                                                  const l1extra::L1HFRings& object) const {
  using namespace l1extra;
  using namespace trigger;

  toc.emplace_back(TriggerL1HfRingEtSums,
                   object.hfEtSum(L1HFRings::kRing1PosEta),
                   object.hfEtSum(L1HFRings::kRing1NegEta),
                   object.hfEtSum(L1HFRings::kRing2PosEta),
                   object.hfEtSum(L1HFRings::kRing2NegEta));
  toc.emplace_back(TriggerL1HfBitCounts,
                   object.hfBitCount(L1HFRings::kRing1PosEta),
                   object.hfBitCount(L1HFRings::kRing1NegEta),
                   object.hfBitCount(L1HFRings::kRing2PosEta),
                   object.hfBitCount(L1HFRings::kRing2NegEta));

  return;
}

void TriggerSummaryProducerAOD::fillTriggerObject(trigger::TriggerObjectCollection& toc,
                                                  const l1extra::L1EtMissParticle& object) const {
  using namespace l1extra;
  using namespace trigger;

  toc.emplace_back(object);
  if (object.type() == L1EtMissParticle::kMET) {
    toc.emplace_back(TriggerL1ETT, object.etTotal(), 0.0, 0.0, 0.0);
  } else if (object.type() == L1EtMissParticle::kMHT) {
    toc.emplace_back(TriggerL1HTT, object.etTotal(), 0.0, 0.0, 0.0);
  } else {
    toc.emplace_back(0, object.etTotal(), 0.0, 0.0, 0.0);
  }

  return;
}

void TriggerSummaryProducerAOD::fillTriggerObject(trigger::TriggerObjectCollection& toc,
                                                  const reco::PFMET& object) const {
  using namespace reco;
  using namespace trigger;

  toc.emplace_back(object);
  toc.emplace_back(TriggerTET, object.sumEt(), 0.0, 0.0, 0.0);
  toc.emplace_back(TriggerMETSig, object.mEtSig(), 0.0, 0.0, 0.0);
  toc.emplace_back(TriggerELongit, object.e_longitudinal(), 0.0, 0.0, 0.0);

  return;
}

void TriggerSummaryProducerAOD::fillTriggerObject(trigger::TriggerObjectCollection& toc,
                                                  const reco::CaloMET& object) const {
  using namespace reco;
  using namespace trigger;

  toc.emplace_back(object);
  toc.emplace_back(TriggerTET, object.sumEt(), 0.0, 0.0, 0.0);
  toc.emplace_back(TriggerMETSig, object.mEtSig(), 0.0, 0.0, 0.0);
  toc.emplace_back(TriggerELongit, object.e_longitudinal(), 0.0, 0.0, 0.0);

  return;
}

void TriggerSummaryProducerAOD::fillTriggerObject(trigger::TriggerObjectCollection& toc,
                                                  const reco::MET& object) const {
  using namespace reco;
  using namespace trigger;

  toc.emplace_back(object);
  toc.emplace_back(TriggerTHT, object.sumEt(), 0.0, 0.0, 0.0);
  toc.emplace_back(TriggerMHTSig, object.mEtSig(), 0.0, 0.0, 0.0);
  toc.emplace_back(TriggerHLongit, object.e_longitudinal(), 0.0, 0.0, 0.0);

  return;
}

template <typename C>
void TriggerSummaryProducerAOD::fillFilterObjectMembers(const edm::Event& iEvent,
                                                        const edm::InputTag& tag,
                                                        const trigger::Vids& ids,
                                                        const std::vector<edm::Ref<C>>& refs,
                                                        const ProductIDtoIndex& offset,
                                                        trigger::Keys& keys,
                                                        trigger::Vids& oIDs) const {
  /// this routine takes a vector of Ref<C>s and determines the
  /// corresponding vector of keys (i.e., indices) into the
  /// TriggerObjectCollection

  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace l1extra;
  using namespace trigger;

  if (ids.size() != refs.size()) {
    LogError("TriggerSummaryProducerAOD") << "Vector length is different: " << ids.size() << " " << refs.size();
  }

  const unsigned int n(min(ids.size(), refs.size()));
  for (unsigned int i = 0; i != n; ++i) {
    const ProductID pid(refs[i].id());
    if (!(pid.isValid())) {
      std::ostringstream ost;
      ost << "Iinvalid pid: " << pid << " FilterTag / Key: " << tag.encode() << " / " << i << "of" << n
          << " CollectionTag / Key: "
          << " <Unrecoverable>"
          << " / " << refs[i].key() << " CollectionType: " << typeid(C).name();
      if (throw_) {
        throw cms::Exception("TriggerSummaryProducerAOD") << ost.str();
      } else {
        LogError("TriggerSummaryProducerAOD") << ost.str();
      }
    } else {
      auto itOffset = offset.find(pid);
      if (itOffset == offset.end()) {
        const string& label(iEvent.getProvenance(pid).moduleLabel());
        const string& instance(iEvent.getProvenance(pid).productInstanceName());
        const string& process(iEvent.getProvenance(pid).processName());
        std::ostringstream ost;
        ost << "Uunknown pid: " << pid << " FilterTag / Key: " << tag.encode() << " / " << i << "of" << n
            << " CollectionTag / Key: " << InputTag(label, instance, process).encode() << " / " << refs[i].key()
            << " CollectionType: " << typeid(C).name();
        if (throw_) {
          throw cms::Exception("TriggerSummaryProducerAOD") << ost.str();
        } else {
          LogError("TriggerSummaryProducerAOD") << ost.str();
        }
      } else {
        fillFilterObjectMember(keys, oIDs, itOffset->second, ids[i], refs[i]);
      }
    }
  }
  return;
}

template <typename C>
void TriggerSummaryProducerAOD::fillFilterObjectMember(
    trigger::Keys& keys, trigger::Vids& ids, const int& offset, const int& id, const edm::Ref<C>& ref) const {
  keys.push_back(offset + ref.key());
  ids.push_back(id);

  return;
}

void TriggerSummaryProducerAOD::fillFilterObjectMember(trigger::Keys& keys,
                                                       trigger::Vids& ids,
                                                       const int& offset,
                                                       const int& id,
                                                       const edm::Ref<l1extra::L1HFRingsCollection>& ref) const {
  using namespace trigger;

  if (id == TriggerL1HfBitCounts) {
    keys.push_back(offset + 2 * ref.key() + 1);
  } else {  // if (ids[i]==TriggerL1HfRingEtSums) {
    keys.push_back(offset + 2 * ref.key() + 0);
  }
  ids.push_back(id);

  return;
}

void TriggerSummaryProducerAOD::fillFilterObjectMember(trigger::Keys& keys,
                                                       trigger::Vids& ids,
                                                       const int& offset,
                                                       const int& id,
                                                       const edm::Ref<l1extra::L1EtMissParticleCollection>& ref) const {
  using namespace trigger;

  if ((id == TriggerL1ETT) || (id == TriggerL1HTT)) {
    keys.push_back(offset + 2 * ref.key() + 1);
  } else {
    keys.push_back(offset + 2 * ref.key() + 0);
  }
  ids.push_back(id);

  return;
}

void TriggerSummaryProducerAOD::fillFilterObjectMember(trigger::Keys& keys,
                                                       trigger::Vids& ids,
                                                       const int& offset,
                                                       const int& id,
                                                       const edm::Ref<reco::PFMETCollection>& ref) const {
  using namespace trigger;

  if ((id == TriggerTHT) || (id == TriggerTET)) {
    keys.push_back(offset + 4 * ref.key() + 1);
  } else if ((id == TriggerMETSig) || (id == TriggerMHTSig)) {
    keys.push_back(offset + 4 * ref.key() + 2);
  } else if ((id == TriggerELongit) || (id == TriggerHLongit)) {
    keys.push_back(offset + 4 * ref.key() + 3);
  } else {
    keys.push_back(offset + 4 * ref.key() + 0);
  }
  ids.push_back(id);

  return;
}

void TriggerSummaryProducerAOD::fillFilterObjectMember(trigger::Keys& keys,
                                                       trigger::Vids& ids,
                                                       const int& offset,
                                                       const int& id,
                                                       const edm::Ref<reco::CaloMETCollection>& ref) const {
  using namespace trigger;

  if ((id == TriggerTHT) || (id == TriggerTET)) {
    keys.push_back(offset + 4 * ref.key() + 1);
  } else if ((id == TriggerMETSig) || (id == TriggerMHTSig)) {
    keys.push_back(offset + 4 * ref.key() + 2);
  } else if ((id == TriggerELongit) || (id == TriggerHLongit)) {
    keys.push_back(offset + 4 * ref.key() + 3);
  } else {
    keys.push_back(offset + 4 * ref.key() + 0);
  }
  ids.push_back(id);

  return;
}

void TriggerSummaryProducerAOD::fillFilterObjectMember(trigger::Keys& keys,
                                                       trigger::Vids& ids,
                                                       const int& offset,
                                                       const int& id,
                                                       const edm::Ref<reco::METCollection>& ref) const {
  using namespace trigger;

  if ((id == TriggerTHT) || (id == TriggerTET)) {
    keys.push_back(offset + 4 * ref.key() + 1);
  } else if ((id == TriggerMETSig) || (id == TriggerMHTSig)) {
    keys.push_back(offset + 4 * ref.key() + 2);
  } else if ((id == TriggerELongit) || (id == TriggerHLongit)) {
    keys.push_back(offset + 4 * ref.key() + 3);
  } else {
    keys.push_back(offset + 4 * ref.key() + 0);
  }
  ids.push_back(id);

  return;
}

void TriggerSummaryProducerAOD::endJob() {
  using namespace std;
  using namespace edm;
  using namespace trigger;

  LogVerbatim("TriggerSummaryProducerAOD") << endl;
  LogVerbatim("TriggerSummaryProducerAOD") << "TriggerSummaryProducerAOD::globalEndJob - accumulated tags:" << endl;

  InputTagSet filterTags(false);
  InputTagSet collectionTags(false);

  filterTags.insert(filterTagsGlobal_.begin(), filterTagsGlobal_.end());
  collectionTags.insert(collectionTagsGlobal_.begin(), collectionTagsGlobal_.end());

  const unsigned int nc(collectionTags.size());
  const unsigned int nf(filterTags.size());
  LogVerbatim("TriggerSummaryProducerAOD") << " Overall number of Collections/Filters: " << nc << "/" << nf << endl;

  LogVerbatim("TriggerSummaryProducerAOD") << " The collections: " << nc << endl;
  const InputTagSet::const_iterator cb(collectionTags.begin());
  const InputTagSet::const_iterator ce(collectionTags.end());
  for (InputTagSet::const_iterator ci = cb; ci != ce; ++ci) {
    LogVerbatim("TriggerSummaryProducerAOD") << "  " << distance(cb, ci) << " " << ci->encode() << endl;
  }

  LogVerbatim("TriggerSummaryProducerAOD") << " The filters:" << nf << endl;
  const InputTagSet::const_iterator fb(filterTags.begin());
  const InputTagSet::const_iterator fe(filterTags.end());
  for (InputTagSet::const_iterator fi = fb; fi != fe; ++fi) {
    LogVerbatim("TriggerSummaryProducerAOD") << "  " << distance(fb, fi) << " " << fi->encode() << endl;
  }

  LogVerbatim("TriggerSummaryProducerAOD") << "TriggerSummaryProducerAOD::endJob." << endl;
  LogVerbatim("TriggerSummaryProducerAOD") << endl;

  return;
}

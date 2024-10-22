#include "HLTPixelIsolTrackFilter.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefToBase.h"

#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

HLTPixelIsolTrackFilter::HLTPixelIsolTrackFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig) {
  candTag_ = iConfig.getParameter<edm::InputTag>("candTag");
  hltGTseedlabel_ = iConfig.getParameter<edm::InputTag>("L1GTSeedLabel");
  minDeltaPtL1Jet_ = iConfig.getParameter<double>("MinDeltaPtL1Jet");
  minpttrack_ = iConfig.getParameter<double>("MinPtTrack");
  maxptnearby_ = iConfig.getParameter<double>("MaxPtNearby");
  maxetatrack_ = iConfig.getParameter<double>("MaxEtaTrack");
  minetatrack_ = iConfig.getParameter<double>("MinEtaTrack");
  filterE_ = iConfig.getParameter<bool>("filterTrackEnergy");
  minEnergy_ = iConfig.getParameter<double>("MinEnergyTrack");
  nMaxTrackCandidates_ = iConfig.getParameter<int>("NMaxTrackCandidates");
  dropMultiL2Event_ = iConfig.getParameter<bool>("DropMultiL2Event");
  candToken_ = consumes<reco::IsolatedPixelTrackCandidateCollection>(candTag_);
  hltGTseedToken_ = consumes<trigger::TriggerFilterObjectWithRefs>(hltGTseedlabel_);
}

HLTPixelIsolTrackFilter::~HLTPixelIsolTrackFilter() = default;

void HLTPixelIsolTrackFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("candTag", edm::InputTag("isolPixelTrackProd"));
  desc.add<edm::InputTag>("L1GTSeedLabel", edm::InputTag("hltL1sIsoTrack"));
  desc.add<double>("MaxPtNearby", 2.0);
  desc.add<double>("MinEnergyTrack", 12.0);
  desc.add<double>("MinPtTrack", 3.5);
  desc.add<double>("MaxEtaTrack", 1.15);
  desc.add<double>("MinEtaTrack", 0.0);
  desc.add<double>("MinDeltaPtL1Jet", -40000.0);
  desc.add<bool>("filterTrackEnergy", true);
  desc.add<int>("NMaxTrackCandidates", 10);
  desc.add<bool>("DropMultiL2Event", false);
  descriptions.add("hltPixelIsolTrackFilter", desc);
}

bool HLTPixelIsolTrackFilter::hltFilter(edm::Event& iEvent,
                                        const edm::EventSetup& iSetup,
                                        trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  if (saveTags())
    filterproduct.addCollectionTag(candTag_);

  // Ref to Candidate object to be recorded in filter object
  edm::Ref<reco::IsolatedPixelTrackCandidateCollection> candref;

  // get hold of filtered candidates
  edm::Handle<reco::IsolatedPixelTrackCandidateCollection> recotrackcands;
  iEvent.getByToken(candToken_, recotrackcands);

  //Filtering

  //find leading L1 jet:
  double ptTriggered = -10;

  edm::Handle<trigger::TriggerFilterObjectWithRefs> l1trigobj;
  iEvent.getByToken(hltGTseedToken_, l1trigobj);

  std::vector<edm::Ref<l1extra::L1JetParticleCollection> > l1tauobjref;
  std::vector<edm::Ref<l1extra::L1JetParticleCollection> > l1jetobjref;
  std::vector<edm::Ref<l1extra::L1JetParticleCollection> > l1forjetobjref;

  l1trigobj->getObjects(trigger::TriggerL1TauJet, l1tauobjref);
  l1trigobj->getObjects(trigger::TriggerL1CenJet, l1jetobjref);
  l1trigobj->getObjects(trigger::TriggerL1ForJet, l1forjetobjref);

  for (auto& p : l1tauobjref)
    if (p->pt() > ptTriggered)
      ptTriggered = p->pt();
  for (auto& p : l1jetobjref)
    if (p->pt() > ptTriggered)
      ptTriggered = p->pt();
  for (auto& p : l1forjetobjref)
    if (p->pt() > ptTriggered)
      ptTriggered = p->pt();

  int n = 0;
  for (unsigned int i = 0; i < recotrackcands->size(); i++) {
    candref = edm::Ref<reco::IsolatedPixelTrackCandidateCollection>(recotrackcands, i);

    // cut on deltaPT
    if (ptTriggered - candref->pt() < minDeltaPtL1Jet_)
      continue;

    // select on transverse momentum
    if (!filterE_ && (candref->maxPtPxl() < maxptnearby_) && (candref->pt() > minpttrack_) &&
        fabs(candref->track()->eta()) < maxetatrack_ && fabs(candref->track()->eta()) > minetatrack_) {
      filterproduct.addObject(trigger::TriggerTrack, candref);
      n++;
      LogDebug("IsoTrk") << "PixelIsolP:Candidate[" << n << "] pt|eta|phi " << candref->pt() << "|" << candref->eta()
                         << "|" << candref->phi() << "\n";
    }

    // select on momentum
    if (filterE_) {
      if ((candref->maxPtPxl() < maxptnearby_) && ((candref->pt()) * cosh(candref->track()->eta()) > minEnergy_) &&
          fabs(candref->track()->eta()) < maxetatrack_ && fabs(candref->track()->eta()) > minetatrack_) {
        filterproduct.addObject(trigger::TriggerTrack, candref);
        n++;
        LogDebug("IsoTrk") << "PixelIsolE:Candidate[" << n << "] pt|eta|phi " << candref->pt() << "|" << candref->eta()
                           << "|" << candref->phi() << "\n";
      }
    }

    // stop looping over tracks if max number is reached
    if (!dropMultiL2Event_ && n >= nMaxTrackCandidates_)
      break;

  }  // loop over tracks

  bool accept(n > 0);

  if (dropMultiL2Event_ && n > nMaxTrackCandidates_)
    accept = false;

  return accept;
}

// declare this class as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTPixelIsolTrackFilter);

#include "HLTMultipletFilter.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefToBase.h"

#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <cmath>

HLTMultipletFilter::HLTMultipletFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig) {
  hltEGammaSeedLabel_ = iConfig.getParameter<edm::InputTag>("L1EGammaInputTag");
  hltEtSumSeedLabel_ = iConfig.getParameter<edm::InputTag>("L1EtSumInputTag");
  hltJetSeedLabel_ = iConfig.getParameter<edm::InputTag>("L1JetInputTag");
  hltMuonSeedLabel_ = iConfig.getParameter<edm::InputTag>("L1MuonInputTag");
  hltTauSeedLabel_ = iConfig.getParameter<edm::InputTag>("L1TauInputTag");
  minN_ = iConfig.getParameter<int>("MinN");
  ibxMin_ = iConfig.getParameter<int>("IBxMin");
  ibxMax_ = iConfig.getParameter<int>("IBxMax");
  minEta_ = iConfig.getParameter<double>("MinEta");
  maxEta_ = iConfig.getParameter<double>("MaxEta");
  minPhi_ = iConfig.getParameter<double>("MinPhi");
  maxPhi_ = iConfig.getParameter<double>("MaxPhi");
  minPt_ = iConfig.getParameter<double>("MinPt");

  if (hltEGammaSeedLabel_ == edm::InputTag()) {
    flag_[EGamma] = false;
  } else {
    flag_[EGamma] = true;
    hltEGammaToken_ = consumes<l1t::EGammaBxCollection>(hltEGammaSeedLabel_);
  }
  if (hltEtSumSeedLabel_ == edm::InputTag()) {
    flag_[EtSum] = false;
  } else {
    flag_[EtSum] = true;
    hltEtSumToken_ = consumes<l1t::EtSumBxCollection>(hltEtSumSeedLabel_);
  }
  if (hltJetSeedLabel_ == edm::InputTag()) {
    flag_[Jet] = false;
  } else {
    flag_[Jet] = true;
    hltJetToken_ = consumes<l1t::JetBxCollection>(hltJetSeedLabel_);
  }
  if (hltMuonSeedLabel_ == edm::InputTag()) {
    flag_[Muon] = false;
  } else {
    flag_[Muon] = true;
    hltMuonToken_ = consumes<l1t::MuonBxCollection>(hltMuonSeedLabel_);
  }
  if (hltTauSeedLabel_ == edm::InputTag()) {
    flag_[Tau] = false;
  } else {
    flag_[Tau] = true;
    hltTauToken_ = consumes<l1t::TauBxCollection>(hltTauSeedLabel_);
  }
  edm::LogVerbatim("Report") << "Input Parameters:: minN = " << minN_ << " Bx Range = " << ibxMin_ << ":" << ibxMax_
                             << " minPt = " << minPt_ << " Eta " << minEta_ << ":" << maxEta_ << " Phi " << minPhi_
                             << ":" << maxPhi_ << " GT Seed for EGamma " << hltEGammaSeedLabel_ << ", EtSum "
                             << hltEtSumSeedLabel_ << ", Jet " << hltJetSeedLabel_ << ", Muon " << hltMuonSeedLabel_
                             << ", and Tau " << hltTauSeedLabel_ << std::endl;
}

HLTMultipletFilter::~HLTMultipletFilter() = default;

void HLTMultipletFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("L1EGammaInputTag", edm::InputTag());
  desc.add<edm::InputTag>("L1EtSumInputTag", edm::InputTag());
  desc.add<edm::InputTag>("L1JetInputTag", edm::InputTag("hltCaloStage2Digis:Jet"));
  desc.add<edm::InputTag>("L1MuonInputTag", edm::InputTag());
  desc.add<edm::InputTag>("L1TauInputTag", edm::InputTag("hltCaloStage2Digis:Tau"));
  desc.add<int>("MinN", 1);
  desc.add<int>("IBxMin", 0);
  desc.add<int>("IBxMax", 0);
  desc.add<double>("MinEta", 1.305);
  desc.add<double>("MaxEta", 3.000);
  desc.add<double>("MinPhi", 5.4105);
  desc.add<double>("MaxPhi", 5.5796);
  desc.add<double>("MinPt", 20.0);
  descriptions.add("hltMultipletFilter", desc);
}

bool HLTMultipletFilter::hltFilter(edm::Event& iEvent,
                                   const edm::EventSetup& iSetup,
                                   trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  // the filter object
  if (saveTags()) {
    if (flag_[EGamma])
      filterproduct.addCollectionTag(hltEGammaSeedLabel_);
    if (flag_[EtSum])
      filterproduct.addCollectionTag(hltEtSumSeedLabel_);
    if (flag_[Jet])
      filterproduct.addCollectionTag(hltJetSeedLabel_);
    if (flag_[Muon])
      filterproduct.addCollectionTag(hltMuonSeedLabel_);
    if (flag_[Tau])
      filterproduct.addCollectionTag(hltTauSeedLabel_);
  }

  int nobj(0);

  if (flag_[EGamma]) {
    nobj += objects(iEvent, hltEGammaToken_, hltEGammaSeedLabel_, EGamma);
    if (nobj >= minN_)
      return true;
  }
  if (flag_[EtSum]) {
    nobj += objects(iEvent, hltEtSumToken_, hltEtSumSeedLabel_, EtSum);
    if (nobj >= minN_)
      return true;
  }
  if (flag_[Jet]) {
    nobj += objects(iEvent, hltJetToken_, hltJetSeedLabel_, Jet);
    if (nobj >= minN_)
      return true;
  }
  if (flag_[Muon]) {
    nobj += objects(iEvent, hltMuonToken_, hltMuonSeedLabel_, Muon);
    if (nobj >= minN_)
      return true;
  }
  if (flag_[Tau]) {
    nobj += objects(iEvent, hltTauToken_, hltTauSeedLabel_, Tau);
    if (nobj >= minN_)
      return true;
  }
  return false;
}

template <typename T1>
int HLTMultipletFilter::objects(edm::Event& iEvent,
                                edm::EDGetTokenT<T1> const& hltToken,
                                edm::InputTag const& hltSeedLabel,
                                HLTMultipletFilter::Types type) const {
  int nobj(0);
  edm::Handle<T1> objs;
  iEvent.getByToken(hltToken, objs);
  if (!objs.isValid()) {
    edm::LogWarning("Report") << "Collection with input tag " << hltSeedLabel
                              << " requested, but not found in the event.";
  } else {
    edm::LogVerbatim("Report") << "Collection for type " << type << " has " << objs->size() << " in " << ibxMin_ << ":"
                               << ibxMax_ << " BX's";
    for (int ibx = ibxMin_; ibx <= ibxMax_; ++ibx) {
      for (auto p = objs->begin(ibx); p != objs->end(ibx); ++p) {
        if (p->pt() > minPt_) {
          if (p->eta() > minEta_ && p->eta() < maxEta_) {
            double phi = p->phi();
            if (phi < 0)
              phi += 2 * M_PI;
            if (phi > minPhi_ && phi < maxPhi_)
              ++nobj;
          }
        }
      }
    }
  }
  return nobj;
}

// declare this class as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTMultipletFilter);

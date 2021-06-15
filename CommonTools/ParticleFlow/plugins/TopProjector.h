#ifndef CommonTools_ParticleFlow_TopProjector_
#define CommonTools_ParticleFlow_TopProjector_

// system include files
#include <iostream>
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Provenance/interface/ProductID.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/JetReco/interface/PFJet.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/OverlapChecker.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

/**\class TopProjector
\brief

\author Colin Bernet
\date   february 2008
*/

#include <iostream>

/// This checks a slew of possible overlaps for FwdPtr<Candidate> and derivatives.
template <class Top, class Bottom>
class TopProjectorFwdPtrOverlap {
public:
  typedef edm::FwdPtr<Top> TopFwdPtr;
  typedef edm::FwdPtr<Bottom> BottomFwdPtr;

  explicit TopProjectorFwdPtrOverlap() { bottom_ = 0; }

  explicit TopProjectorFwdPtrOverlap(edm::ParameterSet const& iConfig)
      : bottom_(nullptr), matchByPtrDirect_(iConfig.getParameter<bool>("matchByPtrDirect")) {}

  inline void setBottom(BottomFwdPtr const& bottom) { bottom_ = &bottom; }

  bool operator()(TopFwdPtr const& top) const {
    if (std::is_same<Top, Bottom>::value && matchByPtrDirect_)
      return top.ptr().refCore() == bottom_->ptr().refCore() && top.ptr().key() == bottom_->ptr().key();
    bool topFwdGood = top.ptr().isNonnull() && top.ptr().isAvailable();
    bool topBckGood = top.backPtr().isNonnull() && top.backPtr().isAvailable();
    bool bottomFwdGood = bottom_->ptr().isNonnull() && bottom_->ptr().isAvailable();
    bool bottomBckGood = bottom_->backPtr().isNonnull() && bottom_->backPtr().isAvailable();

    bool matched = (topFwdGood && bottomFwdGood && top.ptr().refCore() == bottom_->ptr().refCore() &&
                    top.ptr().key() == bottom_->ptr().key()) ||
                   (topFwdGood && bottomBckGood && top.ptr().refCore() == bottom_->backPtr().refCore() &&
                    top.ptr().key() == bottom_->backPtr().key()) ||
                   (topBckGood && bottomFwdGood && top.backPtr().refCore() == bottom_->ptr().refCore() &&
                    top.backPtr().key() == bottom_->ptr().key()) ||
                   (topBckGood && bottomBckGood && top.backPtr().refCore() == bottom_->backPtr().refCore() &&
                    top.backPtr().key() == bottom_->backPtr().key());
    if (!matched) {
      for (unsigned isource = 0; isource < top->numberOfSourceCandidatePtrs(); ++isource) {
        reco::CandidatePtr const& topSrcPtr = top->sourceCandidatePtr(isource);
        bool topSrcGood = topSrcPtr.isNonnull() && topSrcPtr.isAvailable();
        if ((topSrcGood && bottomFwdGood && topSrcPtr.refCore() == bottom_->ptr().refCore() &&
             topSrcPtr.key() == bottom_->ptr().key()) ||
            (topSrcGood && bottomBckGood && topSrcPtr.refCore() == bottom_->backPtr().refCore() &&
             topSrcPtr.key() == bottom_->backPtr().key())) {
          matched = true;
          break;
        }
      }
    }
    if (!matched) {
      for (unsigned isource = 0; isource < (*bottom_)->numberOfSourceCandidatePtrs(); ++isource) {
        reco::CandidatePtr const& bottomSrcPtr = (*bottom_)->sourceCandidatePtr(isource);
        bool bottomSrcGood = bottomSrcPtr.isNonnull() && bottomSrcPtr.isAvailable();
        if ((topFwdGood && bottomSrcGood && bottomSrcPtr.refCore() == top.ptr().refCore() &&
             bottomSrcPtr.key() == top.ptr().key()) ||
            (topBckGood && bottomSrcGood && bottomSrcPtr.refCore() == top.backPtr().refCore() &&
             bottomSrcPtr.key() == top.backPtr().key())) {
          matched = true;
          break;
        }
      }
    }

    return matched;
  }

protected:
  BottomFwdPtr const* bottom_;
  const bool matchByPtrDirect_ = false;
};

/// This checks matching based on delta R
template <class Top, class Bottom>
class TopProjectorDeltaROverlap {
public:
  typedef edm::FwdPtr<Top> TopFwdPtr;
  typedef edm::FwdPtr<Bottom> BottomFwdPtr;

  explicit TopProjectorDeltaROverlap() { bottom_ = 0; }
  explicit TopProjectorDeltaROverlap(edm::ParameterSet const& config)
      : deltaR2_(config.getParameter<double>("deltaR")),
        bottom_(nullptr),
        bottomCPtr_(nullptr),
        botEta_(-999.f),
        botPhi_(0.f) {
    deltaR2_ *= deltaR2_;
  }

  inline void setBottom(BottomFwdPtr const& bottom) {
    bottom_ = &bottom;
    bottomCPtr_ = &**bottom_;
    botEta_ = bottomCPtr_->eta();
    botPhi_ = bottomCPtr_->phi();
  }

  bool operator()(TopFwdPtr const& top) const {
    const Top& oTop = *top;
    float topEta = oTop.eta();
    float topPhi = oTop.phi();
    bool matched = reco::deltaR2(topEta, topPhi, botEta_, botPhi_) < deltaR2_;
    return matched;
  }

protected:
  double deltaR2_;
  BottomFwdPtr const* bottom_;
  const Bottom* bottomCPtr_;
  float botEta_, botPhi_;
};

template <class Top, class Bottom, class Matcher = TopProjectorFwdPtrOverlap<Top, Bottom>>
class TopProjector : public edm::stream::EDProducer<> {
public:
  typedef std::vector<Top> TopCollection;
  typedef edm::FwdPtr<Top> TopFwdPtr;
  typedef std::vector<TopFwdPtr> TopFwdPtrCollection;

  typedef std::vector<Bottom> BottomCollection;
  typedef edm::FwdPtr<Bottom> BottomFwdPtr;
  typedef std::vector<BottomFwdPtr> BottomFwdPtrCollection;

  TopProjector(const edm::ParameterSet&);

  ~TopProjector() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  /// Matching method.
  Matcher match_;

  /// enable? if not, all candidates in the bottom collection are copied to the output collection
  const bool enable_;

  /// name of the top projection
  const std::string name_;

  /// input tag for the top (masking) collection
  const edm::EDGetTokenT<TopFwdPtrCollection> tokenTop_;

  /// input tag for the masked collection.
  const edm::EDGetTokenT<BottomFwdPtrCollection> tokenBottom_;
};

template <class Top, class Bottom, class Matcher>
TopProjector<Top, Bottom, Matcher>::TopProjector(const edm::ParameterSet& iConfig)
    : match_(iConfig),
      enable_(iConfig.getParameter<bool>("enable")),
      name_(iConfig.getUntrackedParameter<std::string>("name", "No Name")),
      tokenTop_(consumes<TopFwdPtrCollection>(iConfig.getParameter<edm::InputTag>("topCollection"))),
      tokenBottom_(consumes<BottomFwdPtrCollection>(iConfig.getParameter<edm::InputTag>("bottomCollection"))) {
  // will produce a collection of the unmasked candidates in the
  // bottom collection
  produces<BottomFwdPtrCollection>();
}

template <class Top, class Bottom, class Matcher>
void TopProjector<Top, Bottom, Matcher>::fillDescriptions(edm::ConfigurationDescriptions& desc) {
  edm::ParameterSetDescription psD;
  psD.add<bool>("enable");
  if (std::is_same<Matcher, TopProjectorDeltaROverlap<Top, Bottom>>::value)
    psD.add<double>("deltaR");
  psD.addUntracked<std::string>("name", "No Name");
  psD.add<edm::InputTag>("topCollection");
  psD.add<edm::InputTag>("bottomCollection");
  if (std::is_same<Matcher, TopProjectorFwdPtrOverlap<Top, Bottom>>::value)
    psD.add<bool>("matchByPtrDirect", false)->setComment("fast check by ptr() only");
  desc.addWithDefaultLabel(psD);
}

template <class Top, class Bottom, class Matcher>
void TopProjector<Top, Bottom, Matcher>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // get the various collections

  // Access the masking collection
  auto const& tops = iEvent.get(tokenTop_);
  std::list<TopFwdPtr> topsList;

  for (auto const& top : tops) {
    topsList.push_back(top);
  }

  // Access the collection to
  // be masked by the other ones
  auto const& bottoms = iEvent.get(tokenBottom_);

  // output collection of FwdPtrs to objects,
  // selected from the Bottom collection
  std::unique_ptr<BottomFwdPtrCollection> pBottomFwdPtrOutput(new BottomFwdPtrCollection);

  LogDebug("TopProjection") << " Remaining candidates in the bottom collection ------ ";

  int iB = -1;
  for (auto const& bottom : bottoms) {
    iB++;
    match_.setBottom(bottom);
    auto found = topsList.end();
    if (enable_) {
      found = std::find_if(topsList.begin(), topsList.end(), match_);
    }

    // If this is masked in the top projection, we remove it.
    if (found != topsList.end()) {
      LogDebug("TopProjection") << "X " << iB << *bottom;
      topsList.erase(found);
      continue;
    }
    // otherwise, we keep it.
    else {
      LogDebug("TopProjection") << "O " << iB << *bottom;
      pBottomFwdPtrOutput->push_back(bottom);
    }
  }

  iEvent.put(std::move(pBottomFwdPtrOutput));
}

#endif

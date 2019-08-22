//****************************************************************
//
// A simple class to combine two candidate collections into a single
// collection of ptrs to the candidates
// note: it is a std::vector as the candidates are from different
// collections
//
// collection 1 is added fully while collection 2 is deltaR cross
// cleaned against collection 2
//
// usecase: getting a unified list of e/gammas + jets for jet core
//          regional tracking
//
//****************************************************************

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Math/interface/deltaR.h"

class CandMergerCleanOthersByDR : public edm::global::EDProducer<> {
public:
  explicit CandMergerCleanOthersByDR(const edm::ParameterSet&);
  ~CandMergerCleanOthersByDR() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  const edm::EDGetTokenT<edm::View<reco::Candidate>> coll1Token_;
  const edm::EDGetTokenT<edm::View<reco::Candidate>> coll2Token_;
  const float maxDR2ToClean_;
};

namespace {
  double pow2(double val) { return val * val; }
}  // namespace

CandMergerCleanOthersByDR::CandMergerCleanOthersByDR(const edm::ParameterSet& iConfig)
    : coll1Token_(consumes<edm::View<reco::Candidate>>(iConfig.getParameter<edm::InputTag>("coll1"))),
      coll2Token_(consumes<edm::View<reco::Candidate>>(iConfig.getParameter<edm::InputTag>("coll2"))),
      maxDR2ToClean_(pow2(iConfig.getParameter<double>("maxDRToClean"))) {
  produces<std::vector<edm::Ptr<reco::Candidate>>>();
}

namespace {
  template <typename T>
  edm::Handle<T> getHandle(const edm::Event& iEvent, const edm::EDGetTokenT<T>& token) {
    edm::Handle<T> handle;
    iEvent.getByToken(token, handle);
    return handle;
  }

  bool hasDRMatch(const reco::Candidate& cand,
                  const std::vector<std::pair<float, float>>& etaPhisToMatch,
                  const float maxDR2) {
    const float candEta = cand.eta();
    const float candPhi = cand.phi();
    for (const auto& etaPhi : etaPhisToMatch) {
      if (reco::deltaR2(candEta, candPhi, etaPhi.first, etaPhi.second) <= maxDR2) {
        return true;
      }
    }
    return false;
  }
}  // namespace

// ------------ method called to produce the data  ------------
void CandMergerCleanOthersByDR::produce(edm::StreamID streamID,
                                        edm::Event& iEvent,
                                        const edm::EventSetup& iSetup) const {
  auto outColl = std::make_unique<std::vector<edm::Ptr<reco::Candidate>>>();

  auto coll1Handle = getHandle(iEvent, coll1Token_);
  auto coll2Handle = getHandle(iEvent, coll2Token_);

  std::vector<std::pair<float, float>> coll1EtaPhis;
  for (size_t objNr = 0; objNr < coll1Handle->size(); objNr++) {
    edm::Ptr<reco::Candidate> objPtr(coll1Handle, objNr);
    coll1EtaPhis.push_back({objPtr->eta(), objPtr->phi()});  //just to speed up the DR match
    outColl->push_back(objPtr);
  }
  for (size_t objNr = 0; objNr < coll2Handle->size(); objNr++) {
    edm::Ptr<reco::Candidate> objPtr(coll2Handle, objNr);
    if (!hasDRMatch(*objPtr, coll1EtaPhis, maxDR2ToClean_)) {
      outColl->push_back(objPtr);
    }
  }

  iEvent.put(std::move(outColl));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void CandMergerCleanOthersByDR::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("coll1", edm::InputTag("egammasForCoreTracking"));
  desc.add<edm::InputTag>("coll2", edm::InputTag("jetsForCoreTracking"));
  desc.add<double>("maxDRToClean", 0.05);
  descriptions.add("candMergerCleanOthersByDR", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(CandMergerCleanOthersByDR);

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"

namespace pat {
  class PATJetCandidatesRekeyer : public edm::stream::EDProducer<> {
  public:
    explicit PATJetCandidatesRekeyer(const edm::ParameterSet &iConfig);
    ~PATJetCandidatesRekeyer() override;

    void produce(edm::Event &, const edm::EventSetup &) override;

  private:
    // configurables
    edm::EDGetTokenT<std::vector<pat::Jet>> src_;
    edm::EDGetTokenT<reco::CandidateView> pcNewCandViewToken_;
    edm::EDGetTokenT<pat::PackedCandidateCollection> pcNewToken_;
    // std::string subjetLabel_;
  };
}  // namespace pat

using namespace pat;

PATJetCandidatesRekeyer::PATJetCandidatesRekeyer(const edm::ParameterSet &iConfig)
    : src_(consumes<std::vector<pat::Jet>>(iConfig.getParameter<edm::InputTag>("src"))),
      pcNewCandViewToken_(consumes<reco::CandidateView>(iConfig.getParameter<edm::InputTag>("packedPFCandidatesNew"))),
      pcNewToken_(
          consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("packedPFCandidatesNew"))) {
  produces<std::vector<pat::Jet>>();
  produces<edm::OwnVector<reco::BaseTagInfo>>("tagInfos");
}

PATJetCandidatesRekeyer::~PATJetCandidatesRekeyer() {}

void PATJetCandidatesRekeyer::produce(edm::Event &iEvent, edm::EventSetup const &) {
  edm::Handle<std::vector<pat::Jet>> src;
  iEvent.getByToken(src_, src);

  edm::Handle<reco::CandidateView> pcNewCandViewHandle;
  iEvent.getByToken(pcNewCandViewToken_, pcNewCandViewHandle);

  edm::Handle<pat::PackedCandidateCollection> pcNewHandle;
  iEvent.getByToken(pcNewToken_, pcNewHandle);

  edm::RefProd<edm::OwnVector<reco::BaseTagInfo>> h_tagInfosOut =
      iEvent.getRefBeforePut<edm::OwnVector<reco::BaseTagInfo>>("tagInfos");
  auto tagInfosOut = std::make_unique<edm::OwnVector<reco::BaseTagInfo>>();

  auto outPtrP = std::make_unique<std::vector<pat::Jet>>();
  outPtrP->reserve(src->size());

  //
  //
  //
  for (const auto &jet : *src) {
    for (const auto &info : jet.tagInfosFwdPtr()) {
      tagInfosOut->push_back(*info);
    }
  }

  edm::OrphanHandle<edm::OwnVector<reco::BaseTagInfo>> oh_tagInfosOut = iEvent.put(std::move(tagInfosOut), "tagInfos");

  //
  //
  //
  unsigned int tagInfoIndex = 0;

  for (const auto &obj : *src) {
    // copy original pat object and append to vector
    outPtrP->emplace_back(obj);

    reco::CompositePtrCandidate::daughters old = outPtrP->back().daughterPtrVector();
    outPtrP->back().clearDaughters();
    for (const auto &dauItr : old) {
      outPtrP->back().addDaughter(edm::Ptr<reco::Candidate>(pcNewHandle, dauItr.key()));
    }

    // Copy the tag infos
    for (TagInfoFwdPtrCollection::const_iterator iinfoBegin = outPtrP->back().tagInfosFwdPtr().begin(),
                                                 iinfoEnd = outPtrP->back().tagInfosFwdPtr().end(),
                                                 iinfo = iinfoBegin;
         iinfo != iinfoEnd;
         ++iinfo) {
      // Update the "forward" bit of the FwdPtr to point at the new collection.
      // ptr to "this" info in the global list
      edm::Ptr<reco::BaseTagInfo> outPtr(oh_tagInfosOut, tagInfoIndex);
      outPtrP->back().updateFwdTagInfoFwdPtr(iinfo - iinfoBegin, outPtr);
      ++tagInfoIndex;
    }
  }

  iEvent.put(std::move(outPtrP));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATJetCandidatesRekeyer);

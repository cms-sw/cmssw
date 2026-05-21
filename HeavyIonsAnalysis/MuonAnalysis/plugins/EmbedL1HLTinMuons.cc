#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "Geometry/CommonTopologies/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"

namespace pat {

  class EmbedL1HLTinMuons : public edm::global::EDProducer<> {
  public:
    explicit EmbedL1HLTinMuons(const edm::ParameterSet& iConfig)
        : muonToken_(consumes<pat::MuonCollection>(iConfig.getParameter<edm::InputTag>("muons"))),
          triggerResultsToken_(consumes<edm::TriggerResults>(iConfig.getParameter<edm::InputTag>("triggerResults"))),
          triggerObjectsToken_(
              consumes<pat::TriggerObjectStandAloneCollection>(iConfig.getParameter<edm::InputTag>("triggerObjects"))),
          geometryToken_(esConsumes()) {
      produces<pat::MuonCollection>();
    }
    ~EmbedL1HLTinMuons() override{};

    void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    const edm::EDGetTokenT<pat::MuonCollection> muonToken_;
    const edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;
    const edm::EDGetTokenT<pat::TriggerObjectStandAloneCollection> triggerObjectsToken_;
    const edm::ESGetToken<GlobalTrackingGeometry, GlobalTrackingGeometryRecord> geometryToken_;

    std::optional<GlobalPoint> getMuonDirection(const reco::MuonChamberMatch&,
                                                const DetId&,
                                                const GlobalTrackingGeometry&) const;

    void fillL1TriggerInfo(pat::Muon&,
                           const pat::TriggerObjectStandAloneCollection&,
                           const edm::TriggerNames&,
                           const GlobalTrackingGeometry&) const;

    void fillHLTriggerInfo(pat::Muon&, const pat::TriggerObjectStandAloneCollection&, const edm::TriggerNames&) const;
  };

}  // namespace pat

void pat::EmbedL1HLTinMuons::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  // extract input information
  const auto& muons = iEvent.get(muonToken_);
  const auto& triggerObjects = iEvent.get(triggerObjectsToken_);
  const auto& triggerResults = iEvent.get(triggerResultsToken_);
  const auto& triggerNames = iEvent.triggerNames(triggerResults);
  const auto& geometry = iSetup.getHandle(geometryToken_);

  // initialize output muon collection
  auto output = std::make_unique<pat::MuonCollection>(muons);

  // add trigger information to muons
  for (auto& muon : *output) {
    const_cast<TriggerObjectStandAloneCollection&>(muon.triggerObjectMatches()).clear();
    fillL1TriggerInfo(muon, triggerObjects, triggerNames, *geometry);
    fillHLTriggerInfo(muon, triggerObjects, triggerNames);
  }

  iEvent.put(std::move(output));
}

std::optional<GlobalPoint> pat::EmbedL1HLTinMuons::getMuonDirection(const reco::MuonChamberMatch& chamberMatch,
                                                                    const DetId& chamberId,
                                                                    const GlobalTrackingGeometry& geometry) const {
  const auto& chamberGeometry = geometry.idToDet(chamberId);
  if (chamberGeometry) {
    LocalPoint localPosition(chamberMatch.x, chamberMatch.y, 0);
    return std::optional<GlobalPoint>(std::in_place, chamberGeometry->toGlobal(localPosition));
  }
  return std::optional<GlobalPoint>();
}

void pat::EmbedL1HLTinMuons::fillL1TriggerInfo(pat::Muon& muon,
                                               const pat::TriggerObjectStandAloneCollection& triggerObjects,
                                               const edm::TriggerNames& triggerNames,
                                               const GlobalTrackingGeometry& geometry) const {
  // L1 trigger object parameters are defined at MB2/ME2. Use the muon
  // chamber matching information to get the local direction of the
  // muon trajectory to match the trigger objects
  std::optional<GlobalPoint> muonPosition;
  for (const auto& chamberMatch : muon.matches()) {
    if (chamberMatch.id.subdetId() == MuonSubdetId::DT) {
      DTChamberId detId(chamberMatch.id.rawId());
      if (std::abs(detId.station()) > 3)
        continue;
      muonPosition = getMuonDirection(chamberMatch, detId, geometry);
      if (std::abs(detId.station()) == 2)
        break;
    } else if (chamberMatch.id.subdetId() == MuonSubdetId::CSC) {
      CSCDetId detId(chamberMatch.id.rawId());
      if (std::abs(detId.station()) > 3)
        continue;
      muonPosition = getMuonDirection(chamberMatch, detId, geometry);
      if (std::abs(detId.station()) == 2)
        break;
    }
  }
  if (!muonPosition)
    return;

  // add L1 trigger object to muon
  for (const auto& triggerObject : triggerObjects) {
    if (!triggerObject.hasTriggerObjectType(trigger::TriggerL1Mu))
      continue;
    if (std::abs(triggerObject.eta()) < 0.001) {
      if (std::abs(deltaPhi(triggerObject.phi(), muonPosition->phi())) > 0.1)
        continue;
    } else if (deltaR(triggerObject.p4(), *muonPosition) > 0.15)
      continue;
    muon.addTriggerObjectMatch(triggerObject);
  }
}

void pat::EmbedL1HLTinMuons::fillHLTriggerInfo(pat::Muon& muon,
                                               const pat::TriggerObjectStandAloneCollection& triggerObjects,
                                               const edm::TriggerNames& triggerNames) const {
  // add HLT object to muon
  for (const auto& triggerObject : triggerObjects) {
    if (!triggerObject.hasTriggerObjectType(trigger::TriggerMuon))
      continue;
    if (deltaR(triggerObject.p4(), muon) > 0.1)
      continue;
    muon.addTriggerObjectMatch(triggerObject);
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void pat::EmbedL1HLTinMuons::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("muons", edm::InputTag("unpackedMuons"))->setComment("muon input collection");
  desc.add<edm::InputTag>("triggerResults", edm::InputTag("TriggerResults::HLT"))
      ->setComment("trigger results collection");
  desc.add<edm::InputTag>("triggerObjects", edm::InputTag("slimmedPatTrigger"))
      ->setComment("trigger objects collection");
  descriptions.add("unpackedMuonsWithTrigger", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
using namespace pat;
DEFINE_FWK_MODULE(EmbedL1HLTinMuons);

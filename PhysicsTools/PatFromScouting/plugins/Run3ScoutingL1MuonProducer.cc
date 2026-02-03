/**
 * Run3ScoutingL1MuonProducer
 *
 * Copies L1 muons from gtStage2Digis to produce collections with
 * the standard gmtStage2Digis module label expected by downstream code.
 *
 * This allows scouting MiniAOD to be compatible with standard workflows
 * that expect L1 muons from gmtStage2Digis.
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"

class Run3ScoutingL1MuonProducer : public edm::stream::EDProducer<> {
public:
  explicit Run3ScoutingL1MuonProducer(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  const edm::EDGetTokenT<BXVector<l1t::Muon>> muonToken_;
  const edm::EDPutTokenT<BXVector<l1t::Muon>> muonPutToken_;
};

Run3ScoutingL1MuonProducer::Run3ScoutingL1MuonProducer(const edm::ParameterSet& iConfig)
    : muonToken_(consumes<BXVector<l1t::Muon>>(iConfig.getParameter<edm::InputTag>("muonSource"))),
      muonPutToken_(produces<BXVector<l1t::Muon>>("Muon")) {}

void Run3ScoutingL1MuonProducer::produce(edm::Event& iEvent, const edm::EventSetup&) {
  auto const& muonsIn = iEvent.get(muonToken_);
  BXVector<l1t::Muon> muonsOut;

  muonsOut.setBXRange(muonsIn.getFirstBX(), muonsIn.getLastBX());
  for (int bx = muonsIn.getFirstBX(); bx <= muonsIn.getLastBX(); ++bx) {
    for (auto it = muonsIn.begin(bx); it != muonsIn.end(bx); ++it) {
      muonsOut.push_back(bx, *it);
    }
  }

  iEvent.emplace(muonPutToken_, std::move(muonsOut));
}

void Run3ScoutingL1MuonProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("muonSource", edm::InputTag("gtStage2Digis", "Muon"));
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(Run3ScoutingL1MuonProducer);

#include "L1Trigger/L1TMuonEndCap/plugins/L1TMuonEndCapShowerProducer.h"
#include "L1Trigger/L1TMuonEndCap/interface/Common.h"

L1TMuonEndCapShowerProducer::L1TMuonEndCapShowerProducer(const edm::ParameterSet& iConfig)
    : config_(iConfig),
      tokenCSCShower_(consumes<CSCShowerDigiCollection>(iConfig.getParameter<edm::InputTag>("CSCShowerInput"))),
      sector_processors_shower_() {
  // Make output products
  produces<l1t::RegionalMuonShowerBxCollection>("EMTF");
}

L1TMuonEndCapShowerProducer::~L1TMuonEndCapShowerProducer() {}

void L1TMuonEndCapShowerProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Create pointers to output products
  auto out_showers = std::make_unique<l1t::RegionalMuonShowerBxCollection>();
  out_showers->clear();
  out_showers->setBXRange(0, 0);

  edm::Handle<CSCShowerDigiCollection> showersH;
  iEvent.getByToken(tokenCSCShower_, showersH);
  const CSCShowerDigiCollection& showers = *showersH.product();

  // ___________________________________________________________________________
  // Run the sector processors

  for (int endcap = emtf::MIN_ENDCAP; endcap <= emtf::MAX_ENDCAP; ++endcap) {
    for (int sector = emtf::MIN_TRIGSECTOR; sector <= emtf::MAX_TRIGSECTOR; ++sector) {
      const int es = (endcap - emtf::MIN_ENDCAP) * (emtf::MAX_TRIGSECTOR - emtf::MIN_TRIGSECTOR + 1) +
                     (sector - emtf::MIN_TRIGSECTOR);

      sector_processors_shower_.at(es).configure(config_, endcap, sector);
      sector_processors_shower_.at(es).process(showers, *out_showers);
    }
  }

  // Fill the output products
  iEvent.put(std::move(out_showers), "EMTF");
}

void L1TMuonEndCapShowerProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  // these are different shower selections that can be enabled
  desc.add<bool>("enableOneLooseShower", true);
  desc.add<bool>("enableOneNominalShower", true);
  desc.add<bool>("enableOneTightShower", true);
  desc.add<bool>("enableTwoLooseShowers", false);
  desc.add<edm::InputTag>("CSCShowerInput", edm::InputTag("simCscTriggerPrimitiveDigis"));
  descriptions.add("simEmtfShowersDef", desc);
  descriptions.setComment("This is the generic cfi file for the EMTF shower producer");
}

// Define this as a plug-in
DEFINE_FWK_MODULE(L1TMuonEndCapShowerProducer);

/*
/P.Katsoulis
/G.Karathanasis     
*/

#include "L1TBMTFAlgoSelector.h"

//The Constructor Initializes 2 RegionalMuonCandBxCollections and defines what to be consumed
dqmBmtfAlgoSelector::L1TBMTFAlgoSelector::L1TBMTFAlgoSelector(const edm::ParameterSet & ps)//:
//  bmtfTriggering(new l1t::RegionalMuonCandBxCollection()),
//  bmtfSecondary(new l1t::RegionalMuonCandBxCollection())
{
  bmtfKalmanToken = consumes<l1t::RegionalMuonCandBxCollection>(ps.getParameter<edm::InputTag>("bmtfKalman"));
  bmtfLegacyToken = consumes<l1t::RegionalMuonCandBxCollection>(ps.getParameter<edm::InputTag>("bmtfLegacy"));
  fedToken = consumes<FEDRawDataCollection>(ps.getParameter<edm::InputTag>("feds"));

  produces<l1t::RegionalMuonCandBxCollection>("BMTF");
  produces<l1t::RegionalMuonCandBxCollection>("BMTF2");
}


dqmBmtfAlgoSelector::L1TBMTFAlgoSelector::~L1TBMTFAlgoSelector() {
}

void dqmBmtfAlgoSelector::L1TBMTFAlgoSelector::produce(edm::Event & eve, const edm::EventSetup & eveSetup)
{

  edm::Handle<FEDRawDataCollection> feds;
  eve.getByToken(fedToken, feds);

  //Get the fw-ver
  int nonEmptyFed=0;
  if (feds->FEDData(1376).size() > 0)
    nonEmptyFed = 1376;
  else if (feds->FEDData(1377).size() > 0)
    nonEmptyFed = 1377;
  else {
    edm::LogError("L1TDQM") << "[L1TBMTFAlgoSelector] Both BMTF feds (1376, 1377) seem empty.";
   // return;
  }
  const FEDRawData& l1tRcd = feds->FEDData(nonEmptyFed);;
  edm::LogInfo("L1TDQM") << "L1T Rcd taken from the FEDData.";
  edm::LogInfo("L1TDQM") << "l1tRcd.size=" << l1tRcd.size() << "   for fed:" << nonEmptyFed;

  const unsigned char *data = l1tRcd.data();
  FEDHeader header(data);
  edm::LogInfo("L1TDQM") << "header and data extracted from the Rcd.";

  amc13::Packet packet;
  if (!packet.parse( (const uint64_t*) data, (const uint64_t*) (data + 8),
                     (l1tRcd.size()) / 8,
                     header.lvl1ID(), header.bxID(), false, false ) ) {

    edm::LogError("L1TDQM") << "Could not extract AMC13 Packet.";
   // return;
  }

  edm::LogInfo("L1TDQM") << "AMC13-packet-payload size = " << packet.payload().size();
  unsigned algo_ver;
  if (packet.payload().size() > 0) {
    auto payload64 = ( packet.payload().at(0) ).data();
    const uint32_t *start = (const uint32_t*) payload64.get();
    const uint32_t *end = start + (packet.payload().at(0).size() * 2);

    l1t::MP7Payload payload(start, end, false);
    algo_ver = payload.getAlgorithmFWVersion();

    edm::LogInfo("L1TDQM") << "algo-ver = " << algo_ver << std::endl;
  }
  else {
    edm::LogError("L1TDQM") << "amc13 payload is empty, cannot extract AMC13 Packet...";
    return;
  }

  std::cout << "algo-rev obtained: " << algo_ver << std::endl;

  // Make the Decision which Algo Triggers
  edm::Handle<l1t::RegionalMuonCandBxCollection> bmtfKalman;
  eve.getByToken(bmtfKalmanToken, bmtfKalman);
  edm::Handle<l1t::RegionalMuonCandBxCollection> bmtfLegacy;
  eve.getByToken(bmtfLegacyToken, bmtfLegacy);

  // auto *bmtfKalman_copy = new l1t::RegionalMuonCandBxCollection(bmtfKalman->size(),
  // 								bmtfKalman->getFirstBX(),
  // 								bmtfKalman->getLastBX()
  // 								);
  // auto *bmtfLegacy_copy = new l1t::RegionalMuonCandBxCollection(bmtfLegacy->size(),
  // 								bmtfLegacy->getFirstBX(),
  // 								bmtfLegacy->getLastBX()
  // 								);

  // unsigned int idx = 0;
  // for (auto bx = bmtfKalman_copy->getFirstBX(); bx <= bmtfKalman_copy->getLastBX(); bx++) {

  //   if ( not bmtfKalman->isEmpty(bx) )
  //     bmtfKalman_copy->set(bx, idx, (*bmtfKalman)[idx]);

  //   idx++;
  // }

  // idx = 0;
  // for (auto bx = bmtfLegacy_copy->getFirstBX(); bx <= bmtfLegacy_copy->getLastBX(); bx++) {

  //   if ( not bmtfLegacy->isEmpty(bx) )
  //     bmtfLegacy_copy->set(bx, idx, (*bmtfLegacy)[idx]);

  //   idx++;
  // }

  auto *bmtfKalman_copy = new l1t::RegionalMuonCandBxCollection(*bmtfKalman);
  auto *bmtfLegacy_copy = new l1t::RegionalMuonCandBxCollection(*bmtfLegacy);

  std::cout << "copy RegionalMuonCandBxCollections created" << std::endl;
  std::cout << "bmtfKalman_copy address: " << bmtfKalman_copy << std::endl;
  std::cout << "bmtfLegacy_copy address: " << bmtfLegacy_copy << std::endl;

  std::unique_ptr<l1t::RegionalMuonCandBxCollection> bmtfTriggering, bmtfSecondary;
  if ( algo_ver >= 2499805536) {//95000160(hex)
    // kalman triggers
    bmtfTriggering.reset(bmtfKalman_copy);
    bmtfSecondary.reset(bmtfLegacy_copy);
  }
  else {
    // legacy triggers
    bmtfTriggering.reset(bmtfLegacy_copy);
    bmtfSecondary.reset(bmtfKalman_copy);
  }

  std::cout << "Triggering and Secondary pointers filled:" << std::endl;
  std::cout << "bmtfTriggering address: " << bmtfTriggering.get() << std::endl;
  std::cout << "bmtfSecondary address: " << bmtfSecondary.get() << std::endl;
  
  eve.put(std::move(bmtfTriggering),"BMTF");
  eve.put(std::move(bmtfSecondary),"BMTF2");

  std::cout << "reached return..." << std::endl;
  return;
}

//void L1TBMTFAlgoSelector::beginStream(edm::StreamID) {}
//void L1TBMTFAlgoSelector::endStream() {}

using namespace dqmBmtfAlgoSelector;
DEFINE_FWK_MODULE(L1TBMTFAlgoSelector);

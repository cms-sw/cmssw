/*
/P.Katsoulis
/G.Karathanasis     
*/

#include "L1TBMTFAlgoSelector.h"

//The Constructor defines what to be consumed and produced
dqmBmtfAlgoSelector::L1TBMTFAlgoSelector::L1TBMTFAlgoSelector(const edm::ParameterSet & ps)
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

  //Declare the Handlers
  edm::Handle<FEDRawDataCollection> feds;
  eve.getByToken(fedToken, feds);
  edm::Handle<l1t::RegionalMuonCandBxCollection> bmtfKalman;
  eve.getByToken(bmtfKalmanToken, bmtfKalman);
  edm::Handle<l1t::RegionalMuonCandBxCollection> bmtfLegacy;
  eve.getByToken(bmtfLegacyToken, bmtfLegacy);


  //----> Make RegMuonCand Copies
  auto *bmtfKalman_copy = new l1t::RegionalMuonCandBxCollection(*bmtfKalman);
  auto *bmtfLegacy_copy = new l1t::RegionalMuonCandBxCollection(*bmtfLegacy);
  edm::LogInfo("L1TDQM") << "copy RegionalMuonCandBxCollections created";
  edm::LogInfo("L1TDQM") << "bmtfKalman_copy address: " << bmtfKalman_copy;
  edm::LogInfo("L1TDQM") << "bmtfLegacy_copy address: " << bmtfLegacy_copy;


  //-----> Get the fw-ver
  int nonEmptyFed=0;
  if (feds->FEDData(1376).size() > 0)
    nonEmptyFed = 1376;
  else if (feds->FEDData(1377).size() > 0)
    nonEmptyFed = 1377;
  else {
    edm::LogError("L1TDQM") << "[L1TBMTFAlgoSelector] Both BMTF feds (1376, 1377) seem empty."
			    << "Using Default Configuration."
			    << "Will be wrong if this is Kalman triggering and ZS is enabled";
    //Define the default configuration
    std::unique_ptr<l1t::RegionalMuonCandBxCollection> bmtfTriggering, bmtfSecondary;
    bmtfTriggering.reset(bmtfLegacy_copy);
    bmtfSecondary.reset(bmtfKalman_copy);
    //Print in log the addresses
    edm::LogInfo("L1TDQM") << "Triggering and Secondary pointers filled:";
    edm::LogInfo("L1TDQM") << "bmtfTriggering address: " << bmtfTriggering.get();
    edm::LogInfo("L1TDQM") << "bmtfSecondary address: " << bmtfSecondary.get();
    //Produce the products
    eve.put(std::move(bmtfTriggering),"BMTF");
    eve.put(std::move(bmtfSecondary),"BMTF2");
    return;
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
    return;
  }

  edm::LogInfo("L1TDQM") << "AMC13-packet-payload size = " << packet.payload().size();
  unsigned algo_ver;
  if (!packet.payload().empty()) {
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


  //----->Make the Decision which Algo Triggers
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

  edm::LogInfo("L1TDQM") << "Triggering and Secondary pointers filled:";
  edm::LogInfo("L1TDQM") << "bmtfTriggering address: " << bmtfTriggering.get();
  edm::LogInfo("L1TDQM") << "bmtfSecondary address: " << bmtfSecondary.get();
  
  eve.put(std::move(bmtfTriggering),"BMTF");
  eve.put(std::move(bmtfSecondary),"BMTF2");

  return;
}

using namespace dqmBmtfAlgoSelector;
DEFINE_FWK_MODULE(L1TBMTFAlgoSelector);

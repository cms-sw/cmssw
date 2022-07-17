#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include <iostream>
#include <sstream>

/**
   @class test_PedestalsBuilder 
   @brief Simple class that analyzes Digis produced by RawToDigi unpacker
*/
class test_PedestalsBuilder : public edm::global::EDAnalyzer<> {
public:
  test_PedestalsBuilder(const edm::ParameterSet&) : pedToken_(esConsumes()) {}
  virtual ~test_PedestalsBuilder() override = default;
  void analyze(edm::StreamID, const edm::Event&, const edm::EventSetup&) const override;

private:
  const edm::ESGetToken<SiStripPedestals, SiStripPedestalsRcd> pedToken_;
};

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
void test_PedestalsBuilder::analyze(edm::StreamID, const edm::Event& event, const edm::EventSetup& setup) const {
  LogTrace(mlCabling_) << "[test_PedestalsBuilder::" << __func__ << "]"
                       << " Dumping all FED connections...";

  const SiStripPedestals* peds = &setup.getData(pedToken_);

  // Retrieve DetIds in Pedestals object
  vector<uint32_t> det_ids;
  peds->getDetIds(det_ids);

  // Iterate through DetIds
  vector<uint32_t>::const_iterator det_id = det_ids.begin();
  for (; det_id != det_ids.end(); det_id++) {
    // Retrieve pedestals for given DetId
    SiStripPedestals::Range range = peds->getRange(*det_id);

    // Check if module has 512 or 768 strips (horrible!)
    uint16_t nstrips = 2 * sistrip::STRIPS_PER_FEDCH;
    //     try {
    //       peds->getPed( 2*sistrip::STRIPS_PER_FEDCH, range );
    //     } catch ( cms::Exception& e ) {
    //       nstrips = 2*sistrip::STRIPS_PER_FEDCH;
    //     }

    stringstream ss;
    ss << "[test_PedestalsBuilder::" << __func__ << "]"
       << " Found " << nstrips << " pedestals for DetId " << *det_id << " (ped/low/high): ";

    // Extract peds and low/high thresholds
    for (uint16_t istrip = 0; istrip < nstrips; istrip++) {
      ss << peds->getPed(istrip, range) << ", ";
    }

    LogTrace(mlCabling_) << ss.str();
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(test_PedestalsBuilder);

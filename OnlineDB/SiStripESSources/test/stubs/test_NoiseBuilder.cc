#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>
#include <sstream>

/**
   @class test_NoiseBuilder 
   @brief Simple class that analyzes Digis produced by RawToDigi unpacker
*/
class test_NoiseBuilder : public edm::one::EDAnalyzer<> {
public:
  test_NoiseBuilder(const edm::ParameterSet&) : noiseToken_(esConsumes()) {}
  ~test_NoiseBuilder() override = default;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  const edm::ESGetToken<SiStripNoises, SiStripNoisesRcd> noiseToken_;
};

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
//
void test_NoiseBuilder::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  LogTrace(mlCabling_) << "[test_NoiseBuilder::" << __func__ << "]"
                       << " Dumping all FED connections...";

  const SiStripNoises* noise = &setup.getData(noiseToken_);

  // Retrieve DetIds in Noise object
  vector<uint32_t> det_ids;
  noise->getDetIds(det_ids);

  // Iterate through DetIds
  vector<uint32_t>::const_iterator det_id = det_ids.begin();
  for (; det_id != det_ids.end(); det_id++) {
    // Retrieve noise for given DetId
    SiStripNoises::Range range = noise->getRange(*det_id);

    // Check if module has 512 or 768 strips (horrible!)
    uint16_t nstrips = 2 * sistrip::STRIPS_PER_FEDCH;
    //     try {
    //       noise->getNoise( 2*sistrip::STRIPS_PER_FEDCH, range );
    //     } catch ( cms::Exception& e ) {
    //       nstrips = 2*sistrip::STRIPS_PER_FEDCH;
    //     }

    stringstream ss;
    ss << "[test_NoiseBuilder::" << __func__ << "]"
       << " Found " << nstrips << " noise for DetId " << *det_id << " (noise/disabled): ";

    // Extract noise and low/high thresholds
    for (uint16_t istrip = 0; istrip < nstrips; istrip++) {
      ss << noise->getNoise(istrip, range) << "/"
         << ", ";
    }

    LogTrace(mlCabling_) << ss.str();
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(test_NoiseBuilder);

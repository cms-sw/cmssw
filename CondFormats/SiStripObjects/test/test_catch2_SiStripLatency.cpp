#include <sstream>
#include "catch.hpp"
#include <iostream>
#include <iomanip>  // std::setw

// Include the headers for SiStripLatency and TrackerTopology
#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Utilities/interface/Exception.h"

// Test case to check the SiStripLatency::printSummary and printDebug
TEST_CASE("SiStripLatency basic test", "[SiStripLatency]") {
  // Step 1: Create an empty SiStripLatency object
  SiStripLatency latency;

  // Step 2: Create a mock or dummy TrackerTopology object
  TrackerTopology* trackerTopo = nullptr;  // Assuming null for now (replace with actual initialization if needed)

  // Step 3: Create a stringstream to capture the output
  std::stringstream ssSummary;
  std::stringstream ssDebug;

  // Step 4: Call printSummary and printDebug on the SiStripLatency object
  try {
    latency.printSummary(ssSummary, trackerTopo);
    latency.printDebug(ssDebug, trackerTopo);
  } catch (const cms::Exception& e) {
    FAIL("Exception caught during printSummary or printDebug: " << e.what());
  }

  // Step 5: Optional - Check the output
  REQUIRE(!ssSummary.str().empty());  // Ensure the summary output is not empty
  REQUIRE(!ssDebug.str().empty());    // Ensure the debug output is not empty

  // Print outputs for manual inspection
  std::cout << "Summary Output:\n" << ssSummary.str() << std::endl;
  std::cout << "Debug Output:\n" << ssDebug.str() << std::endl;
}

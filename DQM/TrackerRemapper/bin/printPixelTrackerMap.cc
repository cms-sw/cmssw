#include "DQM/TrackerRemapper/interface/Phase1PixelSummaryMap.h"
#include <iostream>
#include <numeric>  // std::accumulate
#include <iostream>
#include <vector>
#include <sstream>
#include <string>

#include "TCanvas.h"
#include "TStyle.h"

int main(int argc, char* argv[]) {
  gStyle->SetPalette(1);
  // Check if detids are provided as command line arguments
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " detid1 detid2 detid3 ..." << std::endl;
    return 1;
  }

  // Convert command line arguments to a vector of uint32_t detids
  std::vector<uint32_t> detids;
  for (int i = 1; i < argc; ++i) {
    uint32_t detid;
    std::stringstream ss(argv[i]);
    if (!(ss >> detid)) {
      std::cerr << "Error: Invalid detid: " << argv[i] << std::endl;
      return 1;
    }
    detids.push_back(detid);
  }

  // Create the Phase1PixelSummaryMap object
  Phase1PixelSummaryMap theMap("colz", "Marked Pixel Modules", "");
  
  // Create the tracker base map
  theMap.createTrackerBaseMap();

  // Fill the tracker map with provided detids
  for (const auto& detid : detids) {
    theMap.fillTrackerMap(detid, 1.0);
  }

  TCanvas c = TCanvas("c", "c", 3000, 2000);
  theMap.printTrackerMap(c);
  c.SaveAs("Phase1PixelSummaryMap.png");

  std::cout << "Filled tracker map with " << detids.size() << " detids." << std::endl;

  return 0;
}

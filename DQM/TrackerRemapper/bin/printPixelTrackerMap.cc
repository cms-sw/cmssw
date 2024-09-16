#include "DQM/TrackerRemapper/interface/Phase1PixelSummaryMap.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>  // std::accumulate
#include <sstream>
#include <string>
#include <vector>

#include "TCanvas.h"
#include "TStyle.h"

int main(int argc, char* argv[]) {
  std::string inputFile;
  std::vector<std::pair<uint32_t, float>> detidValues;

  // Parse command line arguments
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--input-file" && i + 1 < argc) {
      gStyle->SetPalette(kRainbow);
      gStyle->SetNumberContours(256);
      inputFile = argv[++i];
    } else {
      gStyle->SetPalette(1);
      // Treat as DetId list if no --input-file is provided
      uint32_t detid = std::stoul(argv[i]);
      detidValues.emplace_back(detid, 1.0);  // Default value is 1.0
    }
  }

  // If --input-file is provided, read from file
  if (!inputFile.empty()) {
    std::ifstream file(inputFile);
    if (!file) {
      std::cerr << "Error: Unable to open input file " << inputFile << std::endl;
      return 1;
    }

    std::string line;
    while (std::getline(file, line)) {
      std::istringstream iss(line);
      uint32_t detid;
      float value = 1.0;  // Default value

      iss >> detid;
      if (iss >> value) {  // If a second column exists, read it as value
        detidValues.emplace_back(detid, value);
      } else {
        detidValues.emplace_back(detid, 1.0);
      }
    }
  }

  // Create the map and fill it
  Phase1PixelSummaryMap theMap("colz", "Marked Pixel Modules", "input values");
  theMap.createTrackerBaseMap();

  for (const auto& [detid, value] : detidValues) {
    theMap.fillTrackerMap(detid, value);
  }

  TCanvas c = TCanvas("c", "c", 3000, 2000);
  theMap.printTrackerMap(c);
  c.SaveAs("Phase1PixelSummaryMap.png");

  std::cout << "Filled tracker map with " << detidValues.size() << " detids." << std::endl;

  return 0;
}

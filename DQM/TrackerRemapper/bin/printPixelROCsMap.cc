#include "DQM/TrackerRemapper/interface/Phase1PixelROCMaps.h"
#include <bitset>
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
  std::string inputROCsFile;
  std::vector<std::pair<uint32_t, float>> detidValues;
  std::vector<std::pair<uint32_t, std::bitset<16>>> detidRocs;

  // Parse command line arguments
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--input-file" && i + 1 < argc) {
      gStyle->SetPalette(kRainbow);
      gStyle->SetNumberContours(256);
      inputFile = argv[++i];
    } else if (std::string(argv[i]) == "--input-ROCs" && i + 1 < argc) {
      gStyle->SetPalette(kRainBow);
      gStyle->SetNumberContours(256);
      inputROCsFile = argv[++i];
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

  // If --input-ROCs is provided, read from file
  if (!inputROCsFile.empty()) {
    std::ifstream file(inputROCsFile);
    if (!file) {
      std::cerr << "Error: Unable to open input ROCs file " << inputROCsFile << std::endl;
      return 1;
    }

    std::string line;
    while (std::getline(file, line)) {
      std::istringstream iss(line);
      uint32_t detid;
      std::string rocBits;
      iss >> detid >> rocBits;

      if (rocBits.length() == 16) {
        std::bitset<16> rocs(rocBits);
        detidRocs.emplace_back(detid, rocs);
      } else {
        std::cerr << "Error: Invalid ROC bits string for detid " << detid << std::endl;
        return 1;
      }
    }
  }

  // Create the map and fill it
  Phase1PixelROCMaps theMap("");

  // Fill with detidValues if --input-file or command line DetIds are used
  for (const auto& [detid, value] : detidValues) {
    theMap.fillWholeModule(detid, value);
  }

  // Fill with detidRocs if --input-ROCs is used
  for (const auto& [detid, rocs] : detidRocs) {
    theMap.fillSelectedRocs(detid, rocs, 1.0);  // Default value 1.0
  }

  // Draw and save the map
  TCanvas canvas("Summary", "Summary", 1200, 1600);
  theMap.drawMaps(canvas, "Marked Pixel ROCs");
  canvas.SaveAs("Phase1PixelROCMap.png");

  std::cout << "Filled Phase1 Pixel ROC map with " << detidValues.size() + detidRocs.size() << " detids." << std::endl;

  return 0;
}

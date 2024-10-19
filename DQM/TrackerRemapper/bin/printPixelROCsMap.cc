#include "DQM/TrackerRemapper/interface/Phase1PixelROCMaps.h"
#include <bitset>
#include <cstdint>  // for uint32_t
#include <cstdlib>  // for std::exit
#include <fstream>
#include <iostream>
#include <numeric>  // std::accumulate
#include <sstream>
#include <string>
#include <vector>

#include "TCanvas.h"
#include "TStyle.h"

void showHelp() {
  std::cout << "Usage: \n"
            << "  --input-file <filename>       Specify the input file\n"
            << "  --input-ROCs <filename>       Specify the input ROCs file\n"
            << "  --h or --help                 Show this help message\n"
            << "  <detid>                       Provide DetId (list of DetIds)\n";
}

int main(int argc, char* argv[]) {
  std::string inputFile;
  std::string inputROCsFile;
  std::vector<std::pair<uint32_t, float>> detidValues;
  std::vector<std::pair<uint32_t, std::bitset<16>>> detidRocs;

  // If no arguments are passed or --h/--help is passed, show the help message
  if (argc == 1) {
    showHelp();
    return 0;
  }

  // Parse command line arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--h" || arg == "--help") {
      showHelp();
      return 0;  // Exit after displaying help
    } else if (arg == "--input-file" && i + 1 < argc) {
      inputFile = argv[++i];
    } else if (arg == "--input-ROCs" && i + 1 < argc) {
      inputROCsFile = argv[++i];
    } else {
      // Assume it's a DetId, convert to uint32_t
      try {
        uint32_t detid = std::stoul(arg);
        detidValues.emplace_back(detid, 1.0);  // Default value is 1.0
      } catch (const std::invalid_argument&) {
        std::cerr << "Invalid argument: " << arg << "\n";
        showHelp();
        return 1;
      }
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

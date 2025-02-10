#include "DQM/TrackerRemapper/interface/Phase1PixelSummaryMap.h"
#include <cstdint>  // For uint32_t
#include <cstdlib>  // For std::exit
#include <fstream>
#include <iostream>
#include <numeric>  // std::accumulate
#include <sstream>
#include <string>
#include <vector>

#include "TCanvas.h"
#include "TStyle.h"

void showHelp(const std::string& scriptName) {
  std::cout << "Usage: " << scriptName << " [options] <detid>\n"
            << "  --input-file <filename>       Specify the input file\n"
            << "  --h or --help                 Show this help message\n"
            << "  <detid>                       Provide DetId (list of DetIds)\n";
}

int main(int argc, char* argv[]) {
  std::string inputFile;
  std::vector<std::pair<uint32_t, float>> detidValues;

  // If no arguments are passed or --h/--help is passed, show the help message
  if (argc == 1) {
    showHelp(argv[0]);
    return 0;
  }

  // Parse command line arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--h" || arg == "--help") {
      showHelp(argv[0]);
      return 0;  // Exit after displaying help
    } else if (arg == "--input-file" && i + 1 < argc) {
      gStyle->SetPalette(kRainbow);
      gStyle->SetNumberContours(256);
      inputFile = argv[++i];
    } else {
      gStyle->SetPalette(1);
      // Treat as DetId list if no --input-file is provided
      try {
        uint32_t detid = std::stoul(arg);
        detidValues.emplace_back(detid, 1.0);  // Default value is 1.0
      } catch (const std::invalid_argument&) {
        std::cerr << "Invalid DetId: " << arg << "\n";
        showHelp(argv[0]);
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

#include "CommonTools/MVAUtils/interface/GBRForestTools.h"

#include "TFile.h"

#include <filesystem>
#include <iostream>

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << "Please pass a (gzipped) BDT weight file and a name for the output ROOT file." << std::endl;
    return 1;
  }

  char *inputFileName = argv[1];
  char *outputFileName = argv[2];

  if (!std::filesystem::exists(inputFileName)) {
    std::cout << "Input file " << inputFileName << " does not exists." << std::endl;
    return 1;
  }

  if (std::filesystem::exists(outputFileName)) {
    std::cout << "Output file " << outputFileName << " already exists." << std::endl;
    return 1;
  }

  auto gbrForest = createGBRForest(inputFileName);
  std::cout << "Read GBRForest " << inputFileName << " successfully." << std::endl;

  TFile{outputFileName, "RECREATE"}.WriteObject(gbrForest.get(), "gbrForest");
  std::cout << "GBRForest written to " << outputFileName << " successfully." << std::endl;

  return 0;
}

#include "RecoTauTag/RecoTau/interface/TMVAZipReader.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <stdio.h>
#include <stdlib.h>

// From http://stackoverflow.com/questions/874134/find-if-string-endswith-another-string-in-c
bool hasEnding(std::string const &fullString, std::string const &ending) {
  if (fullString.length() >= ending.length()) {
    return (0 == fullString.compare(
          fullString.length() - ending.length(), ending.length(), ending));
  } else {
    return false;
  }
}

void loadTMVAWeights(TMVA::Reader* reader, const std::string& method,
    const std::string& weightFile, bool verbose) {
  if (verbose)
    std::cout << "Booking TMVA Reader with weight file: " << weightFile
      << std::endl;

  if (hasEnding(weightFile, ".xml")) {
    if (verbose)
      std::cout << "Weight file is pure xml." << std::endl;
    // Let TMVA read the file
    reader->BookMVA(method, weightFile);
  } else if (hasEnding(weightFile, ".gz") || hasEnding(weightFile, ".gzip")) {
    std::string weight_file_name(std::tmpnam(NULL));
    weight_file_name += ".xml";
    if (verbose) {
      std::cout << "Weight file is zipped." << std::endl;
      std::cout << "Unzipping to: " << weight_file_name << std::endl;
    }
    std::string unzipCommand = "gunzip -c " + weightFile + " > " + weight_file_name;
    if (verbose) {
      std::cout << "Running unzip command: " << std::endl;
      std::cout << unzipCommand << std::endl;
    }
    int result = system(unzipCommand.c_str());
    if (result) {
      throw cms::Exception("UnzippingFailed")
        << "I couldn't gunzip " << weightFile << " into tmpfile "
        << weight_file_name << ", sorry." << std::endl;
    }
    if (verbose) {
      std::cout << "Weight file unzipped, booking reader" << std::endl;
    }
    reader->BookMVA(method, weight_file_name);
    if (verbose) {
      std::cout << "Reader booked, deleting file" << std::endl;
    }
    std::string to_delete = "rm " + weight_file_name;
    system(to_delete.c_str());
  } else {
    throw cms::Exception("BadTMVAWeightFilename")
      << "I don't understand the extension on the filename: "
      << weightFile << ", it should be .xml, .gz, or .gzip" << std::endl;
  }
}

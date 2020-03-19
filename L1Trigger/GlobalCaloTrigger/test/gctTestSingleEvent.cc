#include "L1Trigger/GlobalCaloTrigger/test/gctTestSingleEvent.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"

#include <fstream>
#include <iostream>

gctTestSingleEvent::gctTestSingleEvent() {}
gctTestSingleEvent::~gctTestSingleEvent() {}

// Read the region Et values for a single event from a text file and prepare them to be loaded into the GCT
std::vector<L1CaloRegion> gctTestSingleEvent::loadEvent(const std::string &fileName, const int16_t bx) {
  std::vector<L1CaloRegion> result;

  std::ifstream inFile;

  std::cout << "Reading event data from file " << fileName << std::endl;

  inFile.open(fileName.c_str(), std::ios::in);

  unsigned phi, et;

  // Expect each line of the file to start with the phi value, followed by the et
  // for each eta. When we get to the end of the file and try to read the next phi,
  // recognise the end-of-file condition and quit reading.
  inFile >> phi;
  while (!inFile.eof()) {
    for (unsigned eta = 0; eta < L1CaloRegionDetId::N_ETA; ++eta) {
      inFile >> et;
      // Make an input region
      // Arguments to named ctor are (et, overflow, finegrain, mip, quiet, eta, phi)
      L1CaloRegion temp = L1CaloRegion::makeRegionFromGctIndices(et, false, true, false, false, eta, phi);
      temp.setBx(bx);
      result.push_back(temp);
    }
    inFile >> phi;
  }

  // Tidy the input file
  inFile.close();

  // And finish
  return result;
}

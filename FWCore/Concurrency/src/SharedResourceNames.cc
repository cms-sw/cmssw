#include "FWCore/Concurrency/interface/SharedResourceNames.h"

#include <sstream>
#include <atomic>

const std::string edm::SharedResourceNames::kGEANT = "GEANT";
const std::string edm::SharedResourceNames::kCLHEPRandomEngine = "CLHEPRandomEngine";
const std::string edm::SharedResourceNames::kPythia6 = "Pythia6";
const std::string edm::SharedResourceNames::kPythia8 = "Pythia8";
const std::string edm::SharedResourceNames::kPhotos = "Photos";
const std::string edm::SharedResourceNames::kTauola = "Tauola";
const std::string edm::SharedResourceNames::kEvtGen = "EvtGen";

static std::atomic<unsigned int> counter;

// See comments in header file for the purpose of this function.
std::string edm::uniqueSharedResourceName() {
  std::stringstream ss;
  ss << "uniqueSharedResourceName" << counter.fetch_add(1);
  return ss.str();
}

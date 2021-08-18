#include "DataFormats/L1TCalorimeterPhase2/interface/CaloJet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

void l1tp2::CaloJet::warningNoMapping(std::string const& name) {
  edm::LogWarning("CaloJet") << "Error: no mapping for ExperimentalParam: " << name << std::endl;
}

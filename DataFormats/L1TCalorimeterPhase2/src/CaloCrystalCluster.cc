#include "DataFormats/L1TCalorimeterPhase2/interface/CaloCrystalCluster.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

void l1tp2::CaloCrystalCluster::warningNoMapping(const std::string& name) {
  edm::LogWarning("CaloCrystalCluster") << "Error: no mapping for ExperimentalParam: " << name << std::endl;
}

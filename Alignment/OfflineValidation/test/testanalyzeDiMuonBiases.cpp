#include <iostream>
#include <sstream>
#include <TSystem.h>
#include "Alignment/OfflineValidation/macros/analyzeDiMuonBiases.C"

int main(int argc, char** argv) {
  std::cout << "\n==== Executing muon d0 analysis plotting \n" << std::endl;
  analyzeDiMuonBiases("ZmmNtuple_MC_GEN-SIM_null.root", anaKind::d0_t);

  std::cout << "\n==== Executing muon dz analysis plotting \n" << std::endl;
  analyzeDiMuonBiases("ZmmNtuple_MC_GEN-SIM_null.root", anaKind::dz_t);

  std::cout << "\n==== Executing muon sagitta analysis plotting \n" << std::endl;
  analyzeDiMuonBiases("ZmmNtuple_MC_GEN-SIM_null.root", anaKind::sagitta_t);
}

#include <iostream>
#include <sstream>
#include <TSystem.h>
#include "Alignment/OfflineValidation/macros/momentumBiasValidation.C"
#include "Alignment/OfflineValidation/macros/momentumElectronBiasValidation.C"

int main(int argc, char** argv) {
  //std::cout << "\n==== Executing pion analysis plotting \n" << std::endl;
  //eop::momentumBiasValidation("eta", "./", "test_EopTree.root=TestPion");

  std::cout << "\n==== Executing electron analysis plotting \n" << std::endl;
  momentumElectronBiasValidation("eta", "./", "test_EopTreeElectron.root=TestEle", "png", true);
}

#include <iostream>
#include <sstream>
#include "Alignment/OfflineValidation/macros/momentumBiasValidation.C"
#include "Alignment/OfflineValidation/macros/momentumElectronBiasValidation.C"

int main(int argc, char** argv) {
  momentumBiasValidation("eta", "./", "test_EopTree.root=TestPion", true);
  momentumElectronBiasValidation("eta", "./", "test_EopTreeElectron.root=TestEle", "gif", true);
}

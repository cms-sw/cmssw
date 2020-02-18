#include <iostream>
#include <sstream>
#include "Alignment/OfflineValidation/macros/FitPVResiduals.C"
#include "Alignment/OfflineValidation/macros/FitPVResolution.C"

int main(int argc, char** argv) {
  FitPVResiduals("PVValidation_test_0.root=testing", true, true, "", true);
  FitPVResolution("PVValidation_test_0.root=testing", "");
}

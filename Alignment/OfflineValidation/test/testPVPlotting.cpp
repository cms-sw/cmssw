#include <iostream>
#include <sstream>
#include "Alignment/OfflineValidation/interface/FitPVResiduals.h"
#include "Alignment/OfflineValidation/interface/FitPVResolution.h"

int main(int argc, char** argv) {
  FitPVResiduals("PVValidation_test_0.root=testing", true, true, "", true);
  FitPVResolution("PVValidation_test_0.root=testing", "");
}

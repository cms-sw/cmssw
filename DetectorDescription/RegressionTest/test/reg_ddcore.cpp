#include "DetectorDescription/RegressionTest/src/build.h"
#include "DetectorDescription/Core/interface/ClhepEvaluator.h"

int main() {
 ClhepEvaluator eval;
 regressionTest_setup(eval);
 regressionTest_first(eval);
 output("nix");
 return 0;
}

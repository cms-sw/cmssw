#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaTest/interface/alpaka/printAnswer.h"

// each test binary is built for a single Alpaka backend
using namespace ALPAKA_ACCELERATOR_NAMESPACE;
using namespace ALPAKA_ACCELERATOR_NAMESPACE::alpakatest;

int main() { printAnswer(); }

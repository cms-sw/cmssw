#include "DQM/RCTMonitor/interface/RCTMonitor.h"

#include <cmath>

// Define statics for bins etc.
const unsigned int RCTMonitor::ETABINS = 22;
const float RCTMonitor::ETAMIN = -0.5;
const float RCTMonitor::ETAMAX = 21.5;
const unsigned int RCTMonitor::METPHIBINS = 72;
const float RCTMonitor::METPHIMIN = -0.5;
const float RCTMonitor::METPHIMAX = 71.5;
const unsigned int RCTMonitor::PHIBINS = 18;
const float RCTMonitor::PHIMIN = -0.5;
const float RCTMonitor::PHIMAX = 17.5;
const unsigned int RCTMonitor::TPPHIBINS = 72;
const float RCTMonitor::TPPHIMIN = 0.5;
const float RCTMonitor::TPPHIMAX = 72.5;
const unsigned int RCTMonitor::TPETABINS = 65;
const float RCTMonitor::TPETAMIN = -32.5;
const float RCTMonitor::TPETAMAX = 32.5;
const unsigned int RCTMonitor::L1EETABINS = 22;
const float RCTMonitor::L1EETAMIN = -5;
const float RCTMonitor::L1EETAMAX = 5;
const unsigned int RCTMonitor::L1EPHIBINS = 18;
const float RCTMonitor::L1EPHIMIN = -M_PI;
const float RCTMonitor::L1EPHIMAX = M_PI;

// Ranks 6, 10 and 12 bits
const unsigned int RCTMonitor::R6BINS = 64;
const float RCTMonitor::R6MIN = -0.5;
const float RCTMonitor::R6MAX = 63.5;
const unsigned int RCTMonitor::R10BINS = 1024;
const float RCTMonitor::R10MIN = -0.5;
const float RCTMonitor::R10MAX = 1023.5;
const unsigned int RCTMonitor::R12BINS = 4096;
const float RCTMonitor::R12MIN = -0.5;
const float RCTMonitor::R12MAX = 4095.5;

// Normalize Trigger Towers according to physical extent.
const float ScaleINNER = 1.000000;
const float ScaleOUT = 0.423358;
const float ScaleIN = 0.794521;

// Something for the trigger primitives
const unsigned int RCTMonitor::RTPBINS = 101;
const float RCTMonitor::RTPMIN = -0.5;
const float RCTMonitor::RTPMAX = 100.5;

// Physical bins 1 Gev - 1 TeV in 1 GeV steps
const unsigned int RCTMonitor::TEVBINS = 1001;
const float RCTMonitor::TEVMIN = -0.5;
const float RCTMonitor::TEVMAX = 1000.5;

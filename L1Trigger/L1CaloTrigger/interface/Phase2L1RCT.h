#include "L1Trigger/L1CaloTrigger/interface/Phase2L1CaloEGammaEmulator.h"

#ifndef PHASE_2_L1_RCT_H_INCL
#define PHASE_2_L1_RCT_H_INCL

//////////////////////////////////////////////////////////////////////////
// Other emulator helper functions
//////////////////////////////////////////////////////////////////////////

p2eg::ecalRegion_t p2eg::initStructure(p2eg::crystal temporary[p2eg::CRYSTAL_IN_ETA][p2eg::CRYSTAL_IN_PHI]) {
  ap_uint<5> Eta = 0x0;
  ap_uint<5> Phi = 0x0;

  p2eg::ecalRegion_t out;

  out.etaStrip0.cr0.energy = temporary[Eta + 0][Phi + 0].energy;
  out.etaStrip0.cr0.eta = 0;
  out.etaStrip0.cr0.phi = 0;
  out.etaStrip0.cr1.energy = temporary[Eta + 0][Phi + 1].energy;
  out.etaStrip0.cr1.eta = 0;
  out.etaStrip0.cr1.phi = 1;
  out.etaStrip0.cr2.energy = temporary[Eta + 0][Phi + 2].energy;
  out.etaStrip0.cr2.eta = 0;
  out.etaStrip0.cr2.phi = 2;
  out.etaStrip0.cr3.energy = temporary[Eta + 0][Phi + 3].energy;
  out.etaStrip0.cr3.eta = 0;
  out.etaStrip0.cr3.phi = 3;
  out.etaStrip0.cr4.energy = temporary[Eta + 0][Phi + 4].energy;
  out.etaStrip0.cr4.eta = 0;
  out.etaStrip0.cr4.phi = 4;
  out.etaStrip0.cr5.energy = temporary[Eta + 0][Phi + 5].energy;
  out.etaStrip0.cr5.eta = 0;
  out.etaStrip0.cr5.phi = 5;
  out.etaStrip0.cr6.energy = temporary[Eta + 0][Phi + 6].energy;
  out.etaStrip0.cr6.eta = 0;
  out.etaStrip0.cr6.phi = 6;
  out.etaStrip0.cr7.energy = temporary[Eta + 0][Phi + 7].energy;
  out.etaStrip0.cr7.eta = 0;
  out.etaStrip0.cr7.phi = 7;
  out.etaStrip0.cr8.energy = temporary[Eta + 0][Phi + 8].energy;
  out.etaStrip0.cr8.eta = 0;
  out.etaStrip0.cr8.phi = 8;
  out.etaStrip0.cr9.energy = temporary[Eta + 0][Phi + 9].energy;
  out.etaStrip0.cr9.eta = 0;
  out.etaStrip0.cr9.phi = 9;
  out.etaStrip0.cr10.energy = temporary[Eta + 0][Phi + 10].energy;
  out.etaStrip0.cr10.eta = 0;
  out.etaStrip0.cr10.phi = 10;
  out.etaStrip0.cr11.energy = temporary[Eta + 0][Phi + 11].energy;
  out.etaStrip0.cr11.eta = 0;
  out.etaStrip0.cr11.phi = 11;
  out.etaStrip0.cr12.energy = temporary[Eta + 0][Phi + 12].energy;
  out.etaStrip0.cr12.eta = 0;
  out.etaStrip0.cr12.phi = 12;
  out.etaStrip0.cr13.energy = temporary[Eta + 0][Phi + 13].energy;
  out.etaStrip0.cr13.eta = 0;
  out.etaStrip0.cr13.phi = 13;
  out.etaStrip0.cr14.energy = temporary[Eta + 0][Phi + 14].energy;
  out.etaStrip0.cr14.eta = 0;
  out.etaStrip0.cr14.phi = 14;
  out.etaStrip0.cr15.energy = temporary[Eta + 0][Phi + 15].energy;
  out.etaStrip0.cr15.eta = 0;
  out.etaStrip0.cr15.phi = 15;
  out.etaStrip0.cr16.energy = temporary[Eta + 0][Phi + 16].energy;
  out.etaStrip0.cr16.eta = 0;
  out.etaStrip0.cr16.phi = 16;
  out.etaStrip0.cr17.energy = temporary[Eta + 0][Phi + 17].energy;
  out.etaStrip0.cr17.eta = 0;
  out.etaStrip0.cr17.phi = 17;
  out.etaStrip0.cr18.energy = temporary[Eta + 0][Phi + 18].energy;
  out.etaStrip0.cr18.eta = 0;
  out.etaStrip0.cr18.phi = 18;
  out.etaStrip0.cr19.energy = temporary[Eta + 0][Phi + 19].energy;
  out.etaStrip0.cr19.eta = 0;
  out.etaStrip0.cr19.phi = 19;

  out.etaStrip1.cr0.energy = temporary[Eta + 1][Phi + 0].energy;
  out.etaStrip1.cr0.eta = 1;
  out.etaStrip1.cr0.phi = 0;
  out.etaStrip1.cr1.energy = temporary[Eta + 1][Phi + 1].energy;
  out.etaStrip1.cr1.eta = 1;
  out.etaStrip1.cr1.phi = 1;
  out.etaStrip1.cr2.energy = temporary[Eta + 1][Phi + 2].energy;
  out.etaStrip1.cr2.eta = 1;
  out.etaStrip1.cr2.phi = 2;
  out.etaStrip1.cr3.energy = temporary[Eta + 1][Phi + 3].energy;
  out.etaStrip1.cr3.eta = 1;
  out.etaStrip1.cr3.phi = 3;
  out.etaStrip1.cr4.energy = temporary[Eta + 1][Phi + 4].energy;
  out.etaStrip1.cr4.eta = 1;
  out.etaStrip1.cr4.phi = 4;
  out.etaStrip1.cr5.energy = temporary[Eta + 1][Phi + 5].energy;
  out.etaStrip1.cr5.eta = 1;
  out.etaStrip1.cr5.phi = 5;
  out.etaStrip1.cr6.energy = temporary[Eta + 1][Phi + 6].energy;
  out.etaStrip1.cr6.eta = 1;
  out.etaStrip1.cr6.phi = 6;
  out.etaStrip1.cr7.energy = temporary[Eta + 1][Phi + 7].energy;
  out.etaStrip1.cr7.eta = 1;
  out.etaStrip1.cr7.phi = 7;
  out.etaStrip1.cr8.energy = temporary[Eta + 1][Phi + 8].energy;
  out.etaStrip1.cr8.eta = 1;
  out.etaStrip1.cr8.phi = 8;
  out.etaStrip1.cr9.energy = temporary[Eta + 1][Phi + 9].energy;
  out.etaStrip1.cr9.eta = 1;
  out.etaStrip1.cr9.phi = 9;
  out.etaStrip1.cr10.energy = temporary[Eta + 1][Phi + 10].energy;
  out.etaStrip1.cr10.eta = 1;
  out.etaStrip1.cr10.phi = 10;
  out.etaStrip1.cr11.energy = temporary[Eta + 1][Phi + 11].energy;
  out.etaStrip1.cr11.eta = 1;
  out.etaStrip1.cr11.phi = 11;
  out.etaStrip1.cr12.energy = temporary[Eta + 1][Phi + 12].energy;
  out.etaStrip1.cr12.eta = 1;
  out.etaStrip1.cr12.phi = 12;
  out.etaStrip1.cr13.energy = temporary[Eta + 1][Phi + 13].energy;
  out.etaStrip1.cr13.eta = 1;
  out.etaStrip1.cr13.phi = 13;
  out.etaStrip1.cr14.energy = temporary[Eta + 1][Phi + 14].energy;
  out.etaStrip1.cr14.eta = 1;
  out.etaStrip1.cr14.phi = 14;
  out.etaStrip1.cr15.energy = temporary[Eta + 1][Phi + 15].energy;
  out.etaStrip1.cr15.eta = 1;
  out.etaStrip1.cr15.phi = 15;
  out.etaStrip1.cr16.energy = temporary[Eta + 1][Phi + 16].energy;
  out.etaStrip1.cr16.eta = 1;
  out.etaStrip1.cr16.phi = 16;
  out.etaStrip1.cr17.energy = temporary[Eta + 1][Phi + 17].energy;
  out.etaStrip1.cr17.eta = 1;
  out.etaStrip1.cr17.phi = 17;
  out.etaStrip1.cr18.energy = temporary[Eta + 1][Phi + 18].energy;
  out.etaStrip1.cr18.eta = 1;
  out.etaStrip1.cr18.phi = 18;
  out.etaStrip1.cr19.energy = temporary[Eta + 1][Phi + 19].energy;
  out.etaStrip1.cr19.eta = 1;
  out.etaStrip1.cr19.phi = 19;

  out.etaStrip2.cr0.energy = temporary[Eta + 2][Phi + 0].energy;
  out.etaStrip2.cr0.eta = 2;
  out.etaStrip2.cr0.phi = 0;
  out.etaStrip2.cr1.energy = temporary[Eta + 2][Phi + 1].energy;
  out.etaStrip2.cr1.eta = 2;
  out.etaStrip2.cr1.phi = 1;
  out.etaStrip2.cr2.energy = temporary[Eta + 2][Phi + 2].energy;
  out.etaStrip2.cr2.eta = 2;
  out.etaStrip2.cr2.phi = 2;
  out.etaStrip2.cr3.energy = temporary[Eta + 2][Phi + 3].energy;
  out.etaStrip2.cr3.eta = 2;
  out.etaStrip2.cr3.phi = 3;
  out.etaStrip2.cr4.energy = temporary[Eta + 2][Phi + 4].energy;
  out.etaStrip2.cr4.eta = 2;
  out.etaStrip2.cr4.phi = 4;
  out.etaStrip2.cr5.energy = temporary[Eta + 2][Phi + 5].energy;
  out.etaStrip2.cr5.eta = 2;
  out.etaStrip2.cr5.phi = 5;
  out.etaStrip2.cr6.energy = temporary[Eta + 2][Phi + 6].energy;
  out.etaStrip2.cr6.eta = 2;
  out.etaStrip2.cr6.phi = 6;
  out.etaStrip2.cr7.energy = temporary[Eta + 2][Phi + 7].energy;
  out.etaStrip2.cr7.eta = 2;
  out.etaStrip2.cr7.phi = 7;
  out.etaStrip2.cr8.energy = temporary[Eta + 2][Phi + 8].energy;
  out.etaStrip2.cr8.eta = 2;
  out.etaStrip2.cr8.phi = 8;
  out.etaStrip2.cr9.energy = temporary[Eta + 2][Phi + 9].energy;
  out.etaStrip2.cr9.eta = 2;
  out.etaStrip2.cr9.phi = 9;
  out.etaStrip2.cr10.energy = temporary[Eta + 2][Phi + 10].energy;
  out.etaStrip2.cr10.eta = 2;
  out.etaStrip2.cr10.phi = 10;
  out.etaStrip2.cr11.energy = temporary[Eta + 2][Phi + 11].energy;
  out.etaStrip2.cr11.eta = 2;
  out.etaStrip2.cr11.phi = 11;
  out.etaStrip2.cr12.energy = temporary[Eta + 2][Phi + 12].energy;
  out.etaStrip2.cr12.eta = 2;
  out.etaStrip2.cr12.phi = 12;
  out.etaStrip2.cr13.energy = temporary[Eta + 2][Phi + 13].energy;
  out.etaStrip2.cr13.eta = 2;
  out.etaStrip2.cr13.phi = 13;
  out.etaStrip2.cr14.energy = temporary[Eta + 2][Phi + 14].energy;
  out.etaStrip2.cr14.eta = 2;
  out.etaStrip2.cr14.phi = 14;
  out.etaStrip2.cr15.energy = temporary[Eta + 2][Phi + 15].energy;
  out.etaStrip2.cr15.eta = 2;
  out.etaStrip2.cr15.phi = 15;
  out.etaStrip2.cr16.energy = temporary[Eta + 2][Phi + 16].energy;
  out.etaStrip2.cr16.eta = 2;
  out.etaStrip2.cr16.phi = 16;
  out.etaStrip2.cr17.energy = temporary[Eta + 2][Phi + 17].energy;
  out.etaStrip2.cr17.eta = 2;
  out.etaStrip2.cr17.phi = 17;
  out.etaStrip2.cr18.energy = temporary[Eta + 2][Phi + 18].energy;
  out.etaStrip2.cr18.eta = 2;
  out.etaStrip2.cr18.phi = 18;
  out.etaStrip2.cr19.energy = temporary[Eta + 2][Phi + 19].energy;
  out.etaStrip2.cr19.eta = 2;
  out.etaStrip2.cr19.phi = 19;

  out.etaStrip3.cr0.energy = temporary[Eta + 3][Phi + 0].energy;
  out.etaStrip3.cr0.eta = 3;
  out.etaStrip3.cr0.phi = 0;
  out.etaStrip3.cr1.energy = temporary[Eta + 3][Phi + 1].energy;
  out.etaStrip3.cr1.eta = 3;
  out.etaStrip3.cr1.phi = 1;
  out.etaStrip3.cr2.energy = temporary[Eta + 3][Phi + 2].energy;
  out.etaStrip3.cr2.eta = 3;
  out.etaStrip3.cr2.phi = 2;
  out.etaStrip3.cr3.energy = temporary[Eta + 3][Phi + 3].energy;
  out.etaStrip3.cr3.eta = 3;
  out.etaStrip3.cr3.phi = 3;
  out.etaStrip3.cr4.energy = temporary[Eta + 3][Phi + 4].energy;
  out.etaStrip3.cr4.eta = 3;
  out.etaStrip3.cr4.phi = 4;
  out.etaStrip3.cr5.energy = temporary[Eta + 3][Phi + 5].energy;
  out.etaStrip3.cr5.eta = 3;
  out.etaStrip3.cr5.phi = 5;
  out.etaStrip3.cr6.energy = temporary[Eta + 3][Phi + 6].energy;
  out.etaStrip3.cr6.eta = 3;
  out.etaStrip3.cr6.phi = 6;
  out.etaStrip3.cr7.energy = temporary[Eta + 3][Phi + 7].energy;
  out.etaStrip3.cr7.eta = 3;
  out.etaStrip3.cr7.phi = 7;
  out.etaStrip3.cr8.energy = temporary[Eta + 3][Phi + 8].energy;
  out.etaStrip3.cr8.eta = 3;
  out.etaStrip3.cr8.phi = 8;
  out.etaStrip3.cr9.energy = temporary[Eta + 3][Phi + 9].energy;
  out.etaStrip3.cr9.eta = 3;
  out.etaStrip3.cr9.phi = 9;
  out.etaStrip3.cr10.energy = temporary[Eta + 3][Phi + 10].energy;
  out.etaStrip3.cr10.eta = 3;
  out.etaStrip3.cr10.phi = 10;
  out.etaStrip3.cr11.energy = temporary[Eta + 3][Phi + 11].energy;
  out.etaStrip3.cr11.eta = 3;
  out.etaStrip3.cr11.phi = 11;
  out.etaStrip3.cr12.energy = temporary[Eta + 3][Phi + 12].energy;
  out.etaStrip3.cr12.eta = 3;
  out.etaStrip3.cr12.phi = 12;
  out.etaStrip3.cr13.energy = temporary[Eta + 3][Phi + 13].energy;
  out.etaStrip3.cr13.eta = 3;
  out.etaStrip3.cr13.phi = 13;
  out.etaStrip3.cr14.energy = temporary[Eta + 3][Phi + 14].energy;
  out.etaStrip3.cr14.eta = 3;
  out.etaStrip3.cr14.phi = 14;
  out.etaStrip3.cr15.energy = temporary[Eta + 3][Phi + 15].energy;
  out.etaStrip3.cr15.eta = 3;
  out.etaStrip3.cr15.phi = 15;
  out.etaStrip3.cr16.energy = temporary[Eta + 3][Phi + 16].energy;
  out.etaStrip3.cr16.eta = 3;
  out.etaStrip3.cr16.phi = 16;
  out.etaStrip3.cr17.energy = temporary[Eta + 3][Phi + 17].energy;
  out.etaStrip3.cr17.eta = 3;
  out.etaStrip3.cr17.phi = 17;
  out.etaStrip3.cr18.energy = temporary[Eta + 3][Phi + 18].energy;
  out.etaStrip3.cr18.eta = 3;
  out.etaStrip3.cr18.phi = 18;
  out.etaStrip3.cr19.energy = temporary[Eta + 3][Phi + 19].energy;
  out.etaStrip3.cr19.eta = 3;
  out.etaStrip3.cr19.phi = 19;

  out.etaStrip4.cr0.energy = temporary[Eta + 4][Phi + 0].energy;
  out.etaStrip4.cr0.eta = 4;
  out.etaStrip4.cr0.phi = 0;
  out.etaStrip4.cr1.energy = temporary[Eta + 4][Phi + 1].energy;
  out.etaStrip4.cr1.eta = 4;
  out.etaStrip4.cr1.phi = 1;
  out.etaStrip4.cr2.energy = temporary[Eta + 4][Phi + 2].energy;
  out.etaStrip4.cr2.eta = 4;
  out.etaStrip4.cr2.phi = 2;
  out.etaStrip4.cr3.energy = temporary[Eta + 4][Phi + 3].energy;
  out.etaStrip4.cr3.eta = 4;
  out.etaStrip4.cr3.phi = 3;
  out.etaStrip4.cr4.energy = temporary[Eta + 4][Phi + 4].energy;
  out.etaStrip4.cr4.eta = 4;
  out.etaStrip4.cr4.phi = 4;
  out.etaStrip4.cr5.energy = temporary[Eta + 4][Phi + 5].energy;
  out.etaStrip4.cr5.eta = 4;
  out.etaStrip4.cr5.phi = 5;
  out.etaStrip4.cr6.energy = temporary[Eta + 4][Phi + 6].energy;
  out.etaStrip4.cr6.eta = 4;
  out.etaStrip4.cr6.phi = 6;
  out.etaStrip4.cr7.energy = temporary[Eta + 4][Phi + 7].energy;
  out.etaStrip4.cr7.eta = 4;
  out.etaStrip4.cr7.phi = 7;
  out.etaStrip4.cr8.energy = temporary[Eta + 4][Phi + 8].energy;
  out.etaStrip4.cr8.eta = 4;
  out.etaStrip4.cr8.phi = 8;
  out.etaStrip4.cr9.energy = temporary[Eta + 4][Phi + 9].energy;
  out.etaStrip4.cr9.eta = 4;
  out.etaStrip4.cr9.phi = 9;
  out.etaStrip4.cr10.energy = temporary[Eta + 4][Phi + 10].energy;
  out.etaStrip4.cr10.eta = 4;
  out.etaStrip4.cr10.phi = 10;
  out.etaStrip4.cr11.energy = temporary[Eta + 4][Phi + 11].energy;
  out.etaStrip4.cr11.eta = 4;
  out.etaStrip4.cr11.phi = 11;
  out.etaStrip4.cr12.energy = temporary[Eta + 4][Phi + 12].energy;
  out.etaStrip4.cr12.eta = 4;
  out.etaStrip4.cr12.phi = 12;
  out.etaStrip4.cr13.energy = temporary[Eta + 4][Phi + 13].energy;
  out.etaStrip4.cr13.eta = 4;
  out.etaStrip4.cr13.phi = 13;
  out.etaStrip4.cr14.energy = temporary[Eta + 4][Phi + 14].energy;
  out.etaStrip4.cr14.eta = 4;
  out.etaStrip4.cr14.phi = 14;
  out.etaStrip4.cr15.energy = temporary[Eta + 4][Phi + 15].energy;
  out.etaStrip4.cr15.eta = 4;
  out.etaStrip4.cr15.phi = 15;
  out.etaStrip4.cr16.energy = temporary[Eta + 4][Phi + 16].energy;
  out.etaStrip4.cr16.eta = 4;
  out.etaStrip4.cr16.phi = 16;
  out.etaStrip4.cr17.energy = temporary[Eta + 4][Phi + 17].energy;
  out.etaStrip4.cr17.eta = 4;
  out.etaStrip4.cr17.phi = 17;
  out.etaStrip4.cr18.energy = temporary[Eta + 4][Phi + 18].energy;
  out.etaStrip4.cr18.eta = 4;
  out.etaStrip4.cr18.phi = 18;
  out.etaStrip4.cr19.energy = temporary[Eta + 4][Phi + 19].energy;
  out.etaStrip4.cr19.eta = 4;
  out.etaStrip4.cr19.phi = 19;

  out.etaStrip5.cr0.energy = temporary[Eta + 5][Phi + 0].energy;
  out.etaStrip5.cr0.eta = 5;
  out.etaStrip5.cr0.phi = 0;
  out.etaStrip5.cr1.energy = temporary[Eta + 5][Phi + 1].energy;
  out.etaStrip5.cr1.eta = 5;
  out.etaStrip5.cr1.phi = 1;
  out.etaStrip5.cr2.energy = temporary[Eta + 5][Phi + 2].energy;
  out.etaStrip5.cr2.eta = 5;
  out.etaStrip5.cr2.phi = 2;
  out.etaStrip5.cr3.energy = temporary[Eta + 5][Phi + 3].energy;
  out.etaStrip5.cr3.eta = 5;
  out.etaStrip5.cr3.phi = 3;
  out.etaStrip5.cr4.energy = temporary[Eta + 5][Phi + 4].energy;
  out.etaStrip5.cr4.eta = 5;
  out.etaStrip5.cr4.phi = 4;
  out.etaStrip5.cr5.energy = temporary[Eta + 5][Phi + 5].energy;
  out.etaStrip5.cr5.eta = 5;
  out.etaStrip5.cr5.phi = 5;
  out.etaStrip5.cr6.energy = temporary[Eta + 5][Phi + 6].energy;
  out.etaStrip5.cr6.eta = 5;
  out.etaStrip5.cr6.phi = 6;
  out.etaStrip5.cr7.energy = temporary[Eta + 5][Phi + 7].energy;
  out.etaStrip5.cr7.eta = 5;
  out.etaStrip5.cr7.phi = 7;
  out.etaStrip5.cr8.energy = temporary[Eta + 5][Phi + 8].energy;
  out.etaStrip5.cr8.eta = 5;
  out.etaStrip5.cr8.phi = 8;
  out.etaStrip5.cr9.energy = temporary[Eta + 5][Phi + 9].energy;
  out.etaStrip5.cr9.eta = 5;
  out.etaStrip5.cr9.phi = 9;
  out.etaStrip5.cr10.energy = temporary[Eta + 5][Phi + 10].energy;
  out.etaStrip5.cr10.eta = 5;
  out.etaStrip5.cr10.phi = 10;
  out.etaStrip5.cr11.energy = temporary[Eta + 5][Phi + 11].energy;
  out.etaStrip5.cr11.eta = 5;
  out.etaStrip5.cr11.phi = 11;
  out.etaStrip5.cr12.energy = temporary[Eta + 5][Phi + 12].energy;
  out.etaStrip5.cr12.eta = 5;
  out.etaStrip5.cr12.phi = 12;
  out.etaStrip5.cr13.energy = temporary[Eta + 5][Phi + 13].energy;
  out.etaStrip5.cr13.eta = 5;
  out.etaStrip5.cr13.phi = 13;
  out.etaStrip5.cr14.energy = temporary[Eta + 5][Phi + 14].energy;
  out.etaStrip5.cr14.eta = 5;
  out.etaStrip5.cr14.phi = 14;
  out.etaStrip5.cr15.energy = temporary[Eta + 5][Phi + 15].energy;
  out.etaStrip5.cr15.eta = 5;
  out.etaStrip5.cr15.phi = 15;
  out.etaStrip5.cr16.energy = temporary[Eta + 5][Phi + 16].energy;
  out.etaStrip5.cr16.eta = 5;
  out.etaStrip5.cr16.phi = 16;
  out.etaStrip5.cr17.energy = temporary[Eta + 5][Phi + 17].energy;
  out.etaStrip5.cr17.eta = 5;
  out.etaStrip5.cr17.phi = 17;
  out.etaStrip5.cr18.energy = temporary[Eta + 5][Phi + 18].energy;
  out.etaStrip5.cr18.eta = 5;
  out.etaStrip5.cr18.phi = 18;
  out.etaStrip5.cr19.energy = temporary[Eta + 5][Phi + 19].energy;
  out.etaStrip5.cr19.eta = 5;
  out.etaStrip5.cr19.phi = 19;

  out.etaStrip6.cr0.energy = temporary[Eta + 6][Phi + 0].energy;
  out.etaStrip6.cr0.eta = 6;
  out.etaStrip6.cr0.phi = 0;
  out.etaStrip6.cr1.energy = temporary[Eta + 6][Phi + 1].energy;
  out.etaStrip6.cr1.eta = 6;
  out.etaStrip6.cr1.phi = 1;
  out.etaStrip6.cr2.energy = temporary[Eta + 6][Phi + 2].energy;
  out.etaStrip6.cr2.eta = 6;
  out.etaStrip6.cr2.phi = 2;
  out.etaStrip6.cr2.energy = temporary[Eta + 6][Phi + 2].energy;
  out.etaStrip6.cr2.eta = 6;
  out.etaStrip6.cr2.phi = 2;
  out.etaStrip6.cr3.energy = temporary[Eta + 6][Phi + 3].energy;
  out.etaStrip6.cr3.eta = 6;
  out.etaStrip6.cr3.phi = 3;
  out.etaStrip6.cr4.energy = temporary[Eta + 6][Phi + 4].energy;
  out.etaStrip6.cr4.eta = 6;
  out.etaStrip6.cr4.phi = 4;
  out.etaStrip6.cr5.energy = temporary[Eta + 6][Phi + 5].energy;
  out.etaStrip6.cr5.eta = 6;
  out.etaStrip6.cr5.phi = 5;
  out.etaStrip6.cr6.energy = temporary[Eta + 6][Phi + 6].energy;
  out.etaStrip6.cr6.eta = 6;
  out.etaStrip6.cr6.phi = 6;
  out.etaStrip6.cr7.energy = temporary[Eta + 6][Phi + 7].energy;
  out.etaStrip6.cr7.eta = 6;
  out.etaStrip6.cr7.phi = 7;
  out.etaStrip6.cr8.energy = temporary[Eta + 6][Phi + 8].energy;
  out.etaStrip6.cr8.eta = 6;
  out.etaStrip6.cr8.phi = 8;
  out.etaStrip6.cr9.energy = temporary[Eta + 6][Phi + 9].energy;
  out.etaStrip6.cr9.eta = 6;
  out.etaStrip6.cr9.phi = 9;
  out.etaStrip6.cr10.energy = temporary[Eta + 6][Phi + 10].energy;
  out.etaStrip6.cr10.eta = 6;
  out.etaStrip6.cr10.phi = 10;
  out.etaStrip6.cr11.energy = temporary[Eta + 6][Phi + 11].energy;
  out.etaStrip6.cr11.eta = 6;
  out.etaStrip6.cr11.phi = 11;
  out.etaStrip6.cr12.energy = temporary[Eta + 6][Phi + 12].energy;
  out.etaStrip6.cr12.eta = 6;
  out.etaStrip6.cr12.phi = 12;
  out.etaStrip6.cr13.energy = temporary[Eta + 6][Phi + 13].energy;
  out.etaStrip6.cr13.eta = 6;
  out.etaStrip6.cr13.phi = 13;
  out.etaStrip6.cr14.energy = temporary[Eta + 6][Phi + 14].energy;
  out.etaStrip6.cr14.eta = 6;
  out.etaStrip6.cr14.phi = 14;
  out.etaStrip6.cr15.energy = temporary[Eta + 6][Phi + 15].energy;
  out.etaStrip6.cr15.eta = 6;
  out.etaStrip6.cr15.phi = 15;
  out.etaStrip6.cr16.energy = temporary[Eta + 6][Phi + 16].energy;
  out.etaStrip6.cr16.eta = 6;
  out.etaStrip6.cr16.phi = 16;
  out.etaStrip6.cr17.energy = temporary[Eta + 6][Phi + 17].energy;
  out.etaStrip6.cr17.eta = 6;
  out.etaStrip6.cr17.phi = 17;
  out.etaStrip6.cr18.energy = temporary[Eta + 6][Phi + 18].energy;
  out.etaStrip6.cr18.eta = 6;
  out.etaStrip6.cr18.phi = 18;
  out.etaStrip6.cr19.energy = temporary[Eta + 6][Phi + 19].energy;
  out.etaStrip6.cr19.eta = 6;
  out.etaStrip6.cr19.phi = 19;

  out.etaStrip7.cr0.energy = temporary[Eta + 7][Phi + 0].energy;
  out.etaStrip7.cr0.eta = 7;
  out.etaStrip7.cr0.phi = 0;
  out.etaStrip7.cr1.energy = temporary[Eta + 7][Phi + 1].energy;
  out.etaStrip7.cr1.eta = 7;
  out.etaStrip7.cr1.phi = 1;
  out.etaStrip7.cr2.energy = temporary[Eta + 7][Phi + 2].energy;
  out.etaStrip7.cr2.eta = 7;
  out.etaStrip7.cr2.phi = 2;
  out.etaStrip7.cr3.energy = temporary[Eta + 7][Phi + 3].energy;
  out.etaStrip7.cr3.eta = 7;
  out.etaStrip7.cr3.phi = 3;
  out.etaStrip7.cr4.energy = temporary[Eta + 7][Phi + 4].energy;
  out.etaStrip7.cr4.eta = 7;
  out.etaStrip7.cr4.phi = 4;
  out.etaStrip7.cr5.energy = temporary[Eta + 7][Phi + 5].energy;
  out.etaStrip7.cr5.eta = 7;
  out.etaStrip7.cr5.phi = 5;
  out.etaStrip7.cr6.energy = temporary[Eta + 7][Phi + 6].energy;
  out.etaStrip7.cr6.eta = 7;
  out.etaStrip7.cr6.phi = 6;
  out.etaStrip7.cr7.energy = temporary[Eta + 7][Phi + 7].energy;
  out.etaStrip7.cr7.eta = 7;
  out.etaStrip7.cr7.phi = 7;
  out.etaStrip7.cr8.energy = temporary[Eta + 7][Phi + 8].energy;
  out.etaStrip7.cr8.eta = 7;
  out.etaStrip7.cr8.phi = 8;
  out.etaStrip7.cr9.energy = temporary[Eta + 7][Phi + 9].energy;
  out.etaStrip7.cr9.eta = 7;
  out.etaStrip7.cr9.phi = 9;
  out.etaStrip7.cr10.energy = temporary[Eta + 7][Phi + 10].energy;
  out.etaStrip7.cr10.eta = 7;
  out.etaStrip7.cr10.phi = 10;
  out.etaStrip7.cr11.energy = temporary[Eta + 7][Phi + 11].energy;
  out.etaStrip7.cr11.eta = 7;
  out.etaStrip7.cr11.phi = 11;
  out.etaStrip7.cr12.energy = temporary[Eta + 7][Phi + 12].energy;
  out.etaStrip7.cr12.eta = 7;
  out.etaStrip7.cr12.phi = 12;
  out.etaStrip7.cr13.energy = temporary[Eta + 7][Phi + 13].energy;
  out.etaStrip7.cr13.eta = 7;
  out.etaStrip7.cr13.phi = 13;
  out.etaStrip7.cr14.energy = temporary[Eta + 7][Phi + 14].energy;
  out.etaStrip7.cr14.eta = 7;
  out.etaStrip7.cr14.phi = 14;
  out.etaStrip7.cr15.energy = temporary[Eta + 7][Phi + 15].energy;
  out.etaStrip7.cr15.eta = 7;
  out.etaStrip7.cr15.phi = 15;
  out.etaStrip7.cr16.energy = temporary[Eta + 7][Phi + 16].energy;
  out.etaStrip7.cr16.eta = 7;
  out.etaStrip7.cr16.phi = 16;
  out.etaStrip7.cr17.energy = temporary[Eta + 7][Phi + 17].energy;
  out.etaStrip7.cr17.eta = 7;
  out.etaStrip7.cr17.phi = 17;
  out.etaStrip7.cr18.energy = temporary[Eta + 7][Phi + 18].energy;
  out.etaStrip7.cr18.eta = 7;
  out.etaStrip7.cr18.phi = 18;
  out.etaStrip7.cr19.energy = temporary[Eta + 7][Phi + 19].energy;
  out.etaStrip7.cr19.eta = 7;
  out.etaStrip7.cr19.phi = 19;

  out.etaStrip8.cr0.energy = temporary[Eta + 8][Phi + 0].energy;
  out.etaStrip8.cr0.eta = 8;
  out.etaStrip8.cr0.phi = 0;
  out.etaStrip8.cr1.energy = temporary[Eta + 8][Phi + 1].energy;
  out.etaStrip8.cr1.eta = 8;
  out.etaStrip8.cr1.phi = 1;
  out.etaStrip8.cr2.energy = temporary[Eta + 8][Phi + 2].energy;
  out.etaStrip8.cr2.eta = 8;
  out.etaStrip8.cr2.phi = 2;
  out.etaStrip8.cr3.energy = temporary[Eta + 8][Phi + 3].energy;
  out.etaStrip8.cr3.eta = 8;
  out.etaStrip8.cr3.phi = 3;
  out.etaStrip8.cr4.energy = temporary[Eta + 8][Phi + 4].energy;
  out.etaStrip8.cr4.eta = 8;
  out.etaStrip8.cr4.phi = 4;
  out.etaStrip8.cr5.energy = temporary[Eta + 8][Phi + 5].energy;
  out.etaStrip8.cr5.eta = 8;
  out.etaStrip8.cr5.phi = 5;
  out.etaStrip8.cr6.energy = temporary[Eta + 8][Phi + 6].energy;
  out.etaStrip8.cr6.eta = 8;
  out.etaStrip8.cr6.phi = 6;
  out.etaStrip8.cr7.energy = temporary[Eta + 8][Phi + 7].energy;
  out.etaStrip8.cr7.eta = 8;
  out.etaStrip8.cr7.phi = 7;
  out.etaStrip8.cr8.energy = temporary[Eta + 8][Phi + 8].energy;
  out.etaStrip8.cr8.eta = 8;
  out.etaStrip8.cr8.phi = 8;
  out.etaStrip8.cr9.energy = temporary[Eta + 8][Phi + 9].energy;
  out.etaStrip8.cr9.eta = 8;
  out.etaStrip8.cr9.phi = 9;
  out.etaStrip8.cr10.energy = temporary[Eta + 8][Phi + 10].energy;
  out.etaStrip8.cr10.eta = 8;
  out.etaStrip8.cr10.phi = 10;
  out.etaStrip8.cr11.energy = temporary[Eta + 8][Phi + 11].energy;
  out.etaStrip8.cr11.eta = 8;
  out.etaStrip8.cr11.phi = 11;
  out.etaStrip8.cr12.energy = temporary[Eta + 8][Phi + 12].energy;
  out.etaStrip8.cr12.eta = 8;
  out.etaStrip8.cr12.phi = 12;
  out.etaStrip8.cr13.energy = temporary[Eta + 8][Phi + 13].energy;
  out.etaStrip8.cr13.eta = 8;
  out.etaStrip8.cr13.phi = 13;
  out.etaStrip8.cr14.energy = temporary[Eta + 8][Phi + 14].energy;
  out.etaStrip8.cr14.eta = 8;
  out.etaStrip8.cr14.phi = 14;
  out.etaStrip8.cr15.energy = temporary[Eta + 8][Phi + 15].energy;
  out.etaStrip8.cr15.eta = 8;
  out.etaStrip8.cr15.phi = 15;
  out.etaStrip8.cr16.energy = temporary[Eta + 8][Phi + 16].energy;
  out.etaStrip8.cr16.eta = 8;
  out.etaStrip8.cr16.phi = 16;
  out.etaStrip8.cr17.energy = temporary[Eta + 8][Phi + 17].energy;
  out.etaStrip8.cr17.eta = 8;
  out.etaStrip8.cr17.phi = 17;
  out.etaStrip8.cr18.energy = temporary[Eta + 8][Phi + 18].energy;
  out.etaStrip8.cr18.eta = 8;
  out.etaStrip8.cr18.phi = 18;
  out.etaStrip8.cr19.energy = temporary[Eta + 8][Phi + 19].energy;
  out.etaStrip8.cr19.eta = 8;
  out.etaStrip8.cr19.phi = 19;

  out.etaStrip9.cr0.energy = temporary[Eta + 9][Phi + 0].energy;
  out.etaStrip9.cr0.eta = 9;
  out.etaStrip9.cr0.phi = 0;
  out.etaStrip9.cr1.energy = temporary[Eta + 9][Phi + 1].energy;
  out.etaStrip9.cr1.eta = 9;
  out.etaStrip9.cr1.phi = 1;
  out.etaStrip9.cr2.energy = temporary[Eta + 9][Phi + 2].energy;
  out.etaStrip9.cr2.eta = 9;
  out.etaStrip9.cr2.phi = 2;
  out.etaStrip9.cr3.energy = temporary[Eta + 9][Phi + 3].energy;
  out.etaStrip9.cr3.eta = 9;
  out.etaStrip9.cr3.phi = 3;
  out.etaStrip9.cr4.energy = temporary[Eta + 9][Phi + 4].energy;
  out.etaStrip9.cr4.eta = 9;
  out.etaStrip9.cr4.phi = 4;
  out.etaStrip9.cr5.energy = temporary[Eta + 9][Phi + 5].energy;
  out.etaStrip9.cr5.eta = 9;
  out.etaStrip9.cr5.phi = 5;
  out.etaStrip9.cr6.energy = temporary[Eta + 9][Phi + 6].energy;
  out.etaStrip9.cr6.eta = 9;
  out.etaStrip9.cr6.phi = 6;
  out.etaStrip9.cr7.energy = temporary[Eta + 9][Phi + 7].energy;
  out.etaStrip9.cr7.eta = 9;
  out.etaStrip9.cr7.phi = 7;
  out.etaStrip9.cr8.energy = temporary[Eta + 9][Phi + 8].energy;
  out.etaStrip9.cr8.eta = 9;
  out.etaStrip9.cr8.phi = 8;
  out.etaStrip9.cr9.energy = temporary[Eta + 9][Phi + 9].energy;
  out.etaStrip9.cr9.eta = 9;
  out.etaStrip9.cr9.phi = 9;
  out.etaStrip9.cr10.energy = temporary[Eta + 9][Phi + 10].energy;
  out.etaStrip9.cr10.eta = 9;
  out.etaStrip9.cr10.phi = 10;
  out.etaStrip9.cr11.energy = temporary[Eta + 9][Phi + 11].energy;
  out.etaStrip9.cr11.eta = 9;
  out.etaStrip9.cr11.phi = 11;
  out.etaStrip9.cr12.energy = temporary[Eta + 9][Phi + 12].energy;
  out.etaStrip9.cr12.eta = 9;
  out.etaStrip9.cr12.phi = 12;
  out.etaStrip9.cr13.energy = temporary[Eta + 9][Phi + 13].energy;
  out.etaStrip9.cr13.eta = 9;
  out.etaStrip9.cr13.phi = 13;
  out.etaStrip9.cr14.energy = temporary[Eta + 9][Phi + 14].energy;
  out.etaStrip9.cr14.eta = 9;
  out.etaStrip9.cr14.phi = 14;
  out.etaStrip9.cr15.energy = temporary[Eta + 9][Phi + 15].energy;
  out.etaStrip9.cr15.eta = 9;
  out.etaStrip9.cr15.phi = 15;
  out.etaStrip9.cr16.energy = temporary[Eta + 9][Phi + 16].energy;
  out.etaStrip9.cr16.eta = 9;
  out.etaStrip9.cr16.phi = 16;
  out.etaStrip9.cr17.energy = temporary[Eta + 9][Phi + 17].energy;
  out.etaStrip9.cr17.eta = 9;
  out.etaStrip9.cr17.phi = 17;
  out.etaStrip9.cr18.energy = temporary[Eta + 9][Phi + 18].energy;
  out.etaStrip9.cr18.eta = 9;
  out.etaStrip9.cr18.phi = 18;
  out.etaStrip9.cr19.energy = temporary[Eta + 9][Phi + 19].energy;
  out.etaStrip9.cr19.eta = 9;
  out.etaStrip9.cr19.phi = 19;

  out.etaStrip10.cr0.energy = temporary[Eta + 10][Phi + 0].energy;
  out.etaStrip10.cr0.eta = 10;
  out.etaStrip10.cr0.phi = 0;
  out.etaStrip10.cr1.energy = temporary[Eta + 10][Phi + 1].energy;
  out.etaStrip10.cr1.eta = 10;
  out.etaStrip10.cr1.phi = 1;
  out.etaStrip10.cr2.energy = temporary[Eta + 10][Phi + 2].energy;
  out.etaStrip10.cr2.eta = 10;
  out.etaStrip10.cr2.phi = 2;
  out.etaStrip10.cr3.energy = temporary[Eta + 10][Phi + 3].energy;
  out.etaStrip10.cr3.eta = 10;
  out.etaStrip10.cr3.phi = 3;
  out.etaStrip10.cr4.energy = temporary[Eta + 10][Phi + 4].energy;
  out.etaStrip10.cr4.eta = 10;
  out.etaStrip10.cr4.phi = 4;
  out.etaStrip10.cr5.energy = temporary[Eta + 10][Phi + 5].energy;
  out.etaStrip10.cr5.eta = 10;
  out.etaStrip10.cr5.phi = 5;
  out.etaStrip10.cr6.energy = temporary[Eta + 10][Phi + 6].energy;
  out.etaStrip10.cr6.eta = 10;
  out.etaStrip10.cr6.phi = 6;
  out.etaStrip10.cr7.energy = temporary[Eta + 10][Phi + 7].energy;
  out.etaStrip10.cr7.eta = 10;
  out.etaStrip10.cr7.phi = 7;
  out.etaStrip10.cr8.energy = temporary[Eta + 10][Phi + 8].energy;
  out.etaStrip10.cr8.eta = 10;
  out.etaStrip10.cr8.phi = 8;
  out.etaStrip10.cr9.energy = temporary[Eta + 10][Phi + 9].energy;
  out.etaStrip10.cr9.eta = 10;
  out.etaStrip10.cr9.phi = 9;
  out.etaStrip10.cr10.energy = temporary[Eta + 10][Phi + 10].energy;
  out.etaStrip10.cr10.eta = 10;
  out.etaStrip10.cr10.phi = 10;
  out.etaStrip10.cr11.energy = temporary[Eta + 10][Phi + 11].energy;
  out.etaStrip10.cr11.eta = 10;
  out.etaStrip10.cr11.phi = 11;
  out.etaStrip10.cr12.energy = temporary[Eta + 10][Phi + 12].energy;
  out.etaStrip10.cr12.eta = 10;
  out.etaStrip10.cr12.phi = 12;
  out.etaStrip10.cr13.energy = temporary[Eta + 10][Phi + 13].energy;
  out.etaStrip10.cr13.eta = 10;
  out.etaStrip10.cr13.phi = 13;
  out.etaStrip10.cr14.energy = temporary[Eta + 10][Phi + 14].energy;
  out.etaStrip10.cr14.eta = 10;
  out.etaStrip10.cr14.phi = 14;
  out.etaStrip10.cr15.energy = temporary[Eta + 10][Phi + 15].energy;
  out.etaStrip10.cr15.eta = 10;
  out.etaStrip10.cr15.phi = 15;
  out.etaStrip10.cr16.energy = temporary[Eta + 10][Phi + 16].energy;
  out.etaStrip10.cr16.eta = 10;
  out.etaStrip10.cr16.phi = 16;
  out.etaStrip10.cr17.energy = temporary[Eta + 10][Phi + 17].energy;
  out.etaStrip10.cr17.eta = 10;
  out.etaStrip10.cr17.phi = 17;
  out.etaStrip10.cr18.energy = temporary[Eta + 10][Phi + 18].energy;
  out.etaStrip10.cr18.eta = 10;
  out.etaStrip10.cr18.phi = 18;
  out.etaStrip10.cr19.energy = temporary[Eta + 10][Phi + 19].energy;
  out.etaStrip10.cr19.eta = 10;
  out.etaStrip10.cr19.phi = 19;

  out.etaStrip11.cr0.energy = temporary[Eta + 11][Phi + 0].energy;
  out.etaStrip11.cr0.eta = 11;
  out.etaStrip11.cr0.phi = 0;
  out.etaStrip11.cr1.energy = temporary[Eta + 11][Phi + 1].energy;
  out.etaStrip11.cr1.eta = 11;
  out.etaStrip11.cr1.phi = 1;
  out.etaStrip11.cr2.energy = temporary[Eta + 11][Phi + 2].energy;
  out.etaStrip11.cr2.eta = 11;
  out.etaStrip11.cr2.phi = 2;
  out.etaStrip11.cr3.energy = temporary[Eta + 11][Phi + 3].energy;
  out.etaStrip11.cr3.eta = 11;
  out.etaStrip11.cr3.phi = 3;
  out.etaStrip11.cr4.energy = temporary[Eta + 11][Phi + 4].energy;
  out.etaStrip11.cr4.eta = 11;
  out.etaStrip11.cr4.phi = 4;
  out.etaStrip11.cr5.energy = temporary[Eta + 11][Phi + 5].energy;
  out.etaStrip11.cr5.eta = 11;
  out.etaStrip11.cr5.phi = 5;
  out.etaStrip11.cr6.energy = temporary[Eta + 11][Phi + 6].energy;
  out.etaStrip11.cr6.eta = 11;
  out.etaStrip11.cr6.phi = 6;
  out.etaStrip11.cr7.energy = temporary[Eta + 11][Phi + 7].energy;
  out.etaStrip11.cr7.eta = 11;
  out.etaStrip11.cr7.phi = 7;
  out.etaStrip11.cr8.energy = temporary[Eta + 11][Phi + 8].energy;
  out.etaStrip11.cr8.eta = 11;
  out.etaStrip11.cr8.phi = 8;
  out.etaStrip11.cr9.energy = temporary[Eta + 11][Phi + 9].energy;
  out.etaStrip11.cr9.eta = 11;
  out.etaStrip11.cr9.phi = 9;
  out.etaStrip11.cr10.energy = temporary[Eta + 11][Phi + 10].energy;
  out.etaStrip11.cr10.eta = 11;
  out.etaStrip11.cr10.phi = 10;
  out.etaStrip11.cr11.energy = temporary[Eta + 11][Phi + 11].energy;
  out.etaStrip11.cr11.eta = 11;
  out.etaStrip11.cr11.phi = 11;
  out.etaStrip11.cr12.energy = temporary[Eta + 11][Phi + 12].energy;
  out.etaStrip11.cr12.eta = 11;
  out.etaStrip11.cr12.phi = 12;
  out.etaStrip11.cr13.energy = temporary[Eta + 11][Phi + 13].energy;
  out.etaStrip11.cr13.eta = 11;
  out.etaStrip11.cr13.phi = 13;
  out.etaStrip11.cr14.energy = temporary[Eta + 11][Phi + 14].energy;
  out.etaStrip11.cr14.eta = 11;
  out.etaStrip11.cr14.phi = 14;
  out.etaStrip11.cr15.energy = temporary[Eta + 11][Phi + 15].energy;
  out.etaStrip11.cr15.eta = 11;
  out.etaStrip11.cr15.phi = 15;
  out.etaStrip11.cr16.energy = temporary[Eta + 11][Phi + 16].energy;
  out.etaStrip11.cr16.eta = 11;
  out.etaStrip11.cr16.phi = 16;
  out.etaStrip11.cr17.energy = temporary[Eta + 11][Phi + 17].energy;
  out.etaStrip11.cr17.eta = 11;
  out.etaStrip11.cr17.phi = 17;
  out.etaStrip11.cr18.energy = temporary[Eta + 11][Phi + 18].energy;
  out.etaStrip11.cr18.eta = 11;
  out.etaStrip11.cr18.phi = 18;
  out.etaStrip11.cr19.energy = temporary[Eta + 11][Phi + 19].energy;
  out.etaStrip11.cr19.eta = 11;
  out.etaStrip11.cr19.phi = 19;

  out.etaStrip12.cr0.energy = temporary[Eta + 12][Phi + 0].energy;
  out.etaStrip12.cr0.eta = 12;
  out.etaStrip12.cr0.phi = 0;
  out.etaStrip12.cr1.energy = temporary[Eta + 12][Phi + 1].energy;
  out.etaStrip12.cr1.eta = 12;
  out.etaStrip12.cr1.phi = 1;
  out.etaStrip12.cr2.energy = temporary[Eta + 12][Phi + 2].energy;
  out.etaStrip12.cr2.eta = 12;
  out.etaStrip12.cr2.phi = 2;
  out.etaStrip12.cr3.energy = temporary[Eta + 12][Phi + 3].energy;
  out.etaStrip12.cr3.eta = 12;
  out.etaStrip12.cr3.phi = 3;
  out.etaStrip12.cr4.energy = temporary[Eta + 12][Phi + 4].energy;
  out.etaStrip12.cr4.eta = 12;
  out.etaStrip12.cr4.phi = 4;
  out.etaStrip12.cr5.energy = temporary[Eta + 12][Phi + 5].energy;
  out.etaStrip12.cr5.eta = 12;
  out.etaStrip12.cr5.phi = 5;
  out.etaStrip12.cr6.energy = temporary[Eta + 12][Phi + 6].energy;
  out.etaStrip12.cr6.eta = 12;
  out.etaStrip12.cr6.phi = 6;
  out.etaStrip12.cr7.energy = temporary[Eta + 12][Phi + 7].energy;
  out.etaStrip12.cr7.eta = 12;
  out.etaStrip12.cr7.phi = 7;
  out.etaStrip12.cr8.energy = temporary[Eta + 12][Phi + 8].energy;
  out.etaStrip12.cr8.eta = 12;
  out.etaStrip12.cr8.phi = 8;
  out.etaStrip12.cr9.energy = temporary[Eta + 12][Phi + 9].energy;
  out.etaStrip12.cr9.eta = 12;
  out.etaStrip12.cr9.phi = 9;
  out.etaStrip12.cr10.energy = temporary[Eta + 12][Phi + 10].energy;
  out.etaStrip12.cr10.eta = 12;
  out.etaStrip12.cr10.phi = 10;
  out.etaStrip12.cr11.energy = temporary[Eta + 12][Phi + 11].energy;
  out.etaStrip12.cr11.eta = 12;
  out.etaStrip12.cr11.phi = 11;
  out.etaStrip12.cr12.energy = temporary[Eta + 12][Phi + 12].energy;
  out.etaStrip12.cr12.eta = 12;
  out.etaStrip12.cr12.phi = 12;
  out.etaStrip12.cr13.energy = temporary[Eta + 12][Phi + 13].energy;
  out.etaStrip12.cr13.eta = 12;
  out.etaStrip12.cr13.phi = 13;
  out.etaStrip12.cr14.energy = temporary[Eta + 12][Phi + 14].energy;
  out.etaStrip12.cr14.eta = 12;
  out.etaStrip12.cr14.phi = 14;
  out.etaStrip12.cr15.energy = temporary[Eta + 12][Phi + 15].energy;
  out.etaStrip12.cr15.eta = 12;
  out.etaStrip12.cr15.phi = 15;
  out.etaStrip12.cr16.energy = temporary[Eta + 12][Phi + 16].energy;
  out.etaStrip12.cr16.eta = 12;
  out.etaStrip12.cr16.phi = 16;
  out.etaStrip12.cr17.energy = temporary[Eta + 12][Phi + 17].energy;
  out.etaStrip12.cr17.eta = 12;
  out.etaStrip12.cr17.phi = 17;
  out.etaStrip12.cr18.energy = temporary[Eta + 12][Phi + 18].energy;
  out.etaStrip12.cr18.eta = 12;
  out.etaStrip12.cr18.phi = 18;
  out.etaStrip12.cr19.energy = temporary[Eta + 12][Phi + 19].energy;
  out.etaStrip12.cr19.eta = 12;
  out.etaStrip12.cr19.phi = 19;

  out.etaStrip13.cr0.energy = temporary[Eta + 13][Phi + 0].energy;
  out.etaStrip13.cr0.eta = 13;
  out.etaStrip13.cr0.phi = 0;
  out.etaStrip13.cr1.energy = temporary[Eta + 13][Phi + 1].energy;
  out.etaStrip13.cr1.eta = 13;
  out.etaStrip13.cr1.phi = 1;
  out.etaStrip13.cr2.energy = temporary[Eta + 13][Phi + 2].energy;
  out.etaStrip13.cr2.eta = 13;
  out.etaStrip13.cr2.phi = 2;
  out.etaStrip13.cr3.energy = temporary[Eta + 13][Phi + 3].energy;
  out.etaStrip13.cr3.eta = 13;
  out.etaStrip13.cr3.phi = 3;
  out.etaStrip13.cr4.energy = temporary[Eta + 13][Phi + 4].energy;
  out.etaStrip13.cr4.eta = 13;
  out.etaStrip13.cr4.phi = 4;
  out.etaStrip13.cr5.energy = temporary[Eta + 13][Phi + 5].energy;
  out.etaStrip13.cr5.eta = 13;
  out.etaStrip13.cr5.phi = 5;
  out.etaStrip13.cr6.energy = temporary[Eta + 13][Phi + 6].energy;
  out.etaStrip13.cr6.eta = 13;
  out.etaStrip13.cr6.phi = 6;
  out.etaStrip13.cr7.energy = temporary[Eta + 13][Phi + 7].energy;
  out.etaStrip13.cr7.eta = 13;
  out.etaStrip13.cr7.phi = 7;
  out.etaStrip13.cr8.energy = temporary[Eta + 13][Phi + 8].energy;
  out.etaStrip13.cr8.eta = 13;
  out.etaStrip13.cr8.phi = 8;
  out.etaStrip13.cr9.energy = temporary[Eta + 13][Phi + 9].energy;
  out.etaStrip13.cr9.eta = 13;
  out.etaStrip13.cr9.phi = 9;
  out.etaStrip13.cr10.energy = temporary[Eta + 13][Phi + 10].energy;
  out.etaStrip13.cr10.eta = 13;
  out.etaStrip13.cr10.phi = 10;
  out.etaStrip13.cr11.energy = temporary[Eta + 13][Phi + 11].energy;
  out.etaStrip13.cr11.eta = 13;
  out.etaStrip13.cr11.phi = 11;
  out.etaStrip13.cr12.energy = temporary[Eta + 13][Phi + 12].energy;
  out.etaStrip13.cr12.eta = 13;
  out.etaStrip13.cr12.phi = 12;
  out.etaStrip13.cr13.energy = temporary[Eta + 13][Phi + 13].energy;
  out.etaStrip13.cr13.eta = 13;
  out.etaStrip13.cr13.phi = 13;
  out.etaStrip13.cr14.energy = temporary[Eta + 13][Phi + 14].energy;
  out.etaStrip13.cr14.eta = 13;
  out.etaStrip13.cr14.phi = 14;
  out.etaStrip13.cr15.energy = temporary[Eta + 13][Phi + 15].energy;
  out.etaStrip13.cr15.eta = 13;
  out.etaStrip13.cr15.phi = 15;
  out.etaStrip13.cr16.energy = temporary[Eta + 13][Phi + 16].energy;
  out.etaStrip13.cr16.eta = 13;
  out.etaStrip13.cr16.phi = 16;
  out.etaStrip13.cr17.energy = temporary[Eta + 13][Phi + 17].energy;
  out.etaStrip13.cr17.eta = 13;
  out.etaStrip13.cr17.phi = 17;
  out.etaStrip13.cr18.energy = temporary[Eta + 13][Phi + 18].energy;
  out.etaStrip13.cr18.eta = 13;
  out.etaStrip13.cr18.phi = 18;
  out.etaStrip13.cr19.energy = temporary[Eta + 13][Phi + 19].energy;
  out.etaStrip13.cr19.eta = 13;
  out.etaStrip13.cr19.phi = 19;

  out.etaStrip14.cr0.energy = temporary[Eta + 14][Phi + 0].energy;
  out.etaStrip14.cr0.eta = 14;
  out.etaStrip14.cr0.phi = 0;
  out.etaStrip14.cr1.energy = temporary[Eta + 14][Phi + 1].energy;
  out.etaStrip14.cr1.eta = 14;
  out.etaStrip14.cr1.phi = 1;
  out.etaStrip14.cr2.energy = temporary[Eta + 14][Phi + 2].energy;
  out.etaStrip14.cr2.eta = 14;
  out.etaStrip14.cr2.phi = 2;
  out.etaStrip14.cr3.energy = temporary[Eta + 14][Phi + 3].energy;
  out.etaStrip14.cr3.eta = 14;
  out.etaStrip14.cr3.phi = 3;
  out.etaStrip14.cr4.energy = temporary[Eta + 14][Phi + 4].energy;
  out.etaStrip14.cr4.eta = 14;
  out.etaStrip14.cr4.phi = 4;
  out.etaStrip14.cr5.energy = temporary[Eta + 14][Phi + 5].energy;
  out.etaStrip14.cr5.eta = 14;
  out.etaStrip14.cr5.phi = 5;
  out.etaStrip14.cr6.energy = temporary[Eta + 14][Phi + 6].energy;
  out.etaStrip14.cr6.eta = 14;
  out.etaStrip14.cr6.phi = 6;
  out.etaStrip14.cr7.energy = temporary[Eta + 14][Phi + 7].energy;
  out.etaStrip14.cr7.eta = 14;
  out.etaStrip14.cr7.phi = 7;
  out.etaStrip14.cr8.energy = temporary[Eta + 14][Phi + 8].energy;
  out.etaStrip14.cr8.eta = 14;
  out.etaStrip14.cr8.phi = 8;
  out.etaStrip14.cr9.energy = temporary[Eta + 14][Phi + 9].energy;
  out.etaStrip14.cr9.eta = 14;
  out.etaStrip14.cr9.phi = 9;
  out.etaStrip14.cr10.energy = temporary[Eta + 14][Phi + 10].energy;
  out.etaStrip14.cr10.eta = 14;
  out.etaStrip14.cr10.phi = 10;
  out.etaStrip14.cr11.energy = temporary[Eta + 14][Phi + 11].energy;
  out.etaStrip14.cr11.eta = 14;
  out.etaStrip14.cr11.phi = 11;
  out.etaStrip14.cr12.energy = temporary[Eta + 14][Phi + 12].energy;
  out.etaStrip14.cr12.eta = 14;
  out.etaStrip14.cr12.phi = 12;
  out.etaStrip14.cr13.energy = temporary[Eta + 14][Phi + 13].energy;
  out.etaStrip14.cr13.eta = 14;
  out.etaStrip14.cr13.phi = 13;
  out.etaStrip14.cr14.energy = temporary[Eta + 14][Phi + 14].energy;
  out.etaStrip14.cr14.eta = 14;
  out.etaStrip14.cr14.phi = 14;
  out.etaStrip14.cr15.energy = temporary[Eta + 14][Phi + 15].energy;
  out.etaStrip14.cr15.eta = 14;
  out.etaStrip14.cr15.phi = 15;
  out.etaStrip14.cr16.energy = temporary[Eta + 14][Phi + 16].energy;
  out.etaStrip14.cr16.eta = 14;
  out.etaStrip14.cr16.phi = 16;
  out.etaStrip14.cr17.energy = temporary[Eta + 14][Phi + 17].energy;
  out.etaStrip14.cr17.eta = 14;
  out.etaStrip14.cr17.phi = 17;
  out.etaStrip14.cr18.energy = temporary[Eta + 14][Phi + 18].energy;
  out.etaStrip14.cr18.eta = 14;
  out.etaStrip14.cr18.phi = 18;
  out.etaStrip14.cr19.energy = temporary[Eta + 14][Phi + 19].energy;
  out.etaStrip14.cr19.eta = 14;
  out.etaStrip14.cr19.phi = 19;

  return out;
}

//--------------------------------------------------------//

// Compare two ecaltp_t and return the one with the larger pT.
p2eg::ecaltp_t p2eg::bestOf2(const p2eg::ecaltp_t ecaltp0, const p2eg::ecaltp_t ecaltp1) {
  p2eg::ecaltp_t x;
  x = (ecaltp0.energy > ecaltp1.energy) ? ecaltp0 : ecaltp1;

  return x;
}

//--------------------------------------------------------//

// For a given etaStrip_t, find the ecaltp_t (out of 20 of them) with the largest pT, using pairwise comparison
p2eg::ecaltp_t p2eg::getPeakBin20N(const p2eg::etaStrip_t etaStrip) {
  p2eg::ecaltp_t best01 = p2eg::bestOf2(etaStrip.cr0, etaStrip.cr1);
  p2eg::ecaltp_t best23 = p2eg::bestOf2(etaStrip.cr2, etaStrip.cr3);
  p2eg::ecaltp_t best45 = p2eg::bestOf2(etaStrip.cr4, etaStrip.cr5);
  p2eg::ecaltp_t best67 = p2eg::bestOf2(etaStrip.cr6, etaStrip.cr7);
  p2eg::ecaltp_t best89 = p2eg::bestOf2(etaStrip.cr8, etaStrip.cr9);
  p2eg::ecaltp_t best1011 = p2eg::bestOf2(etaStrip.cr10, etaStrip.cr11);
  p2eg::ecaltp_t best1213 = p2eg::bestOf2(etaStrip.cr12, etaStrip.cr13);
  p2eg::ecaltp_t best1415 = p2eg::bestOf2(etaStrip.cr14, etaStrip.cr15);
  p2eg::ecaltp_t best1617 = p2eg::bestOf2(etaStrip.cr16, etaStrip.cr17);
  p2eg::ecaltp_t best1819 = p2eg::bestOf2(etaStrip.cr18, etaStrip.cr19);

  p2eg::ecaltp_t best0123 = p2eg::bestOf2(best01, best23);
  p2eg::ecaltp_t best4567 = p2eg::bestOf2(best45, best67);
  p2eg::ecaltp_t best891011 = p2eg::bestOf2(best89, best1011);
  p2eg::ecaltp_t best12131415 = p2eg::bestOf2(best1213, best1415);
  p2eg::ecaltp_t best16171819 = p2eg::bestOf2(best1617, best1819);

  p2eg::ecaltp_t best01234567 = p2eg::bestOf2(best0123, best4567);
  p2eg::ecaltp_t best89101112131415 = p2eg::bestOf2(best891011, best12131415);

  p2eg::ecaltp_t best0to15 = p2eg::bestOf2(best01234567, best89101112131415);
  p2eg::ecaltp_t bestOf20 = p2eg::bestOf2(best0to15, best16171819);

  return bestOf20;
}

//--------------------------------------------------------//

// For a given etaStripPeak_t (representing the 15 crystals, one per row in eta, not necessarily with the same phi),
// return the crystal with the highest pT).

p2eg::crystalMax p2eg::getPeakBin15N(const p2eg::etaStripPeak_t etaStrip) {
  p2eg::crystalMax x;

  p2eg::ecaltp_t best01 = p2eg::bestOf2(etaStrip.pk0, etaStrip.pk1);
  p2eg::ecaltp_t best23 = p2eg::bestOf2(etaStrip.pk2, etaStrip.pk3);
  p2eg::ecaltp_t best45 = p2eg::bestOf2(etaStrip.pk4, etaStrip.pk5);
  p2eg::ecaltp_t best67 = p2eg::bestOf2(etaStrip.pk6, etaStrip.pk7);
  p2eg::ecaltp_t best89 = p2eg::bestOf2(etaStrip.pk8, etaStrip.pk9);
  p2eg::ecaltp_t best1011 = p2eg::bestOf2(etaStrip.pk10, etaStrip.pk11);
  p2eg::ecaltp_t best1213 = p2eg::bestOf2(etaStrip.pk12, etaStrip.pk13);

  p2eg::ecaltp_t best0123 = p2eg::bestOf2(best01, best23);
  p2eg::ecaltp_t best4567 = p2eg::bestOf2(best45, best67);
  p2eg::ecaltp_t best891011 = p2eg::bestOf2(best89, best1011);
  p2eg::ecaltp_t best121314 = p2eg::bestOf2(best1213, etaStrip.pk14);

  p2eg::ecaltp_t best01234567 = p2eg::bestOf2(best0123, best4567);
  p2eg::ecaltp_t best891011121314 = p2eg::bestOf2(best891011, best121314);

  p2eg::ecaltp_t bestOf15 = p2eg::bestOf2(best01234567, best891011121314);

  x.energy = bestOf15.energy;
  x.etaMax = bestOf15.eta;
  x.phiMax = bestOf15.phi;

  return x;
}

//--------------------------------------------------------//

// Take a 3x4 ECAL region (i.e. 15x20 in crystals, add crystal energies in squares of 5x5, giving
// 3x4 = 12 ECAL tower sums.) Store these 12 values in towerEt.

void p2eg::getECALTowersEt(p2eg::crystal tempX[p2eg::CRYSTAL_IN_ETA][p2eg::CRYSTAL_IN_PHI], ap_uint<12> towerEt[12]) {
  ap_uint<10> temp[p2eg::CRYSTAL_IN_ETA][p2eg::CRYSTAL_IN_PHI];
  ap_uint<12> towerEtN[3][4][5];
  for (int i = 0; i < p2eg::CRYSTAL_IN_ETA; i++) {
    for (int k = 0; k < p2eg::CRYSTAL_IN_PHI; k++) {
      temp[i][k] = tempX[i][k].energy;
    }
  }

  for (int i = 0; i < p2eg::CRYSTAL_IN_ETA; i = i + 5) {
    for (int k = 0; k < p2eg::CRYSTAL_IN_PHI; k = k + 5) {
      towerEtN[i / 5][k / 5][0] = temp[i][k] + temp[i][k + 1] + temp[i][k + 2] + temp[i][k + 3] + temp[i][k + 4];
      towerEtN[i / 5][k / 5][1] =
          temp[i + 1][k] + temp[i + 1][k + 1] + temp[i + 1][k + 2] + temp[i + 1][k + 3] + temp[i + 1][k + 4];
      towerEtN[i / 5][k / 5][2] =
          temp[i + 2][k] + temp[i + 2][k + 1] + temp[i + 2][k + 2] + temp[i + 2][k + 3] + temp[i + 2][k + 4];
      towerEtN[i / 5][k / 5][3] =
          temp[i + 3][k] + temp[i + 3][k + 1] + temp[i + 3][k + 2] + temp[i + 3][k + 3] + temp[i + 3][k + 4];
      towerEtN[i / 5][k / 5][4] =
          temp[i + 4][k] + temp[i + 4][k + 1] + temp[i + 4][k + 2] + temp[i + 4][k + 3] + temp[i + 4][k + 4];
    }
  }

  towerEt[0] = towerEtN[0][0][0] + towerEtN[0][0][1] + towerEtN[0][0][2] + towerEtN[0][0][3] + towerEtN[0][0][4];
  towerEt[1] = towerEtN[0][1][0] + towerEtN[0][1][1] + towerEtN[0][1][2] + towerEtN[0][1][3] + towerEtN[0][1][4];
  towerEt[2] = towerEtN[0][2][0] + towerEtN[0][2][1] + towerEtN[0][2][2] + towerEtN[0][2][3] + towerEtN[0][2][4];
  towerEt[3] = towerEtN[0][3][0] + towerEtN[0][3][1] + towerEtN[0][3][2] + towerEtN[0][3][3] + towerEtN[0][3][4];
  towerEt[4] = towerEtN[1][0][0] + towerEtN[1][0][1] + towerEtN[1][0][2] + towerEtN[1][0][3] + towerEtN[1][0][4];
  towerEt[5] = towerEtN[1][1][0] + towerEtN[1][1][1] + towerEtN[1][1][2] + towerEtN[1][1][3] + towerEtN[1][1][4];
  towerEt[6] = towerEtN[1][2][0] + towerEtN[1][2][1] + towerEtN[1][2][2] + towerEtN[1][2][3] + towerEtN[1][2][4];
  towerEt[7] = towerEtN[1][3][0] + towerEtN[1][3][1] + towerEtN[1][3][2] + towerEtN[1][3][3] + towerEtN[1][3][4];
  towerEt[8] = towerEtN[2][0][0] + towerEtN[2][0][1] + towerEtN[2][0][2] + towerEtN[2][0][3] + towerEtN[2][0][4];
  towerEt[9] = towerEtN[2][1][0] + towerEtN[2][1][1] + towerEtN[2][1][2] + towerEtN[2][1][3] + towerEtN[2][1][4];
  towerEt[10] = towerEtN[2][2][0] + towerEtN[2][2][1] + towerEtN[2][2][2] + towerEtN[2][2][3] + towerEtN[2][2][4];
  towerEt[11] = towerEtN[2][3][0] + towerEtN[2][3][1] + towerEtN[2][3][2] + towerEtN[2][3][3] + towerEtN[2][3][4];

  ap_uint<12> totalEt;
  for (int i = 0; i < 12; i++) {
    totalEt += towerEt[i];
  }
}

//--------------------------------------------------------//

p2eg::clusterInfo p2eg::getClusterPosition(const p2eg::ecalRegion_t ecalRegion) {
  p2eg::etaStripPeak_t etaStripPeak;
  p2eg::clusterInfo cluster;

  etaStripPeak.pk0 = p2eg::getPeakBin20N(ecalRegion.etaStrip0);
  etaStripPeak.pk1 = p2eg::getPeakBin20N(ecalRegion.etaStrip1);
  etaStripPeak.pk2 = p2eg::getPeakBin20N(ecalRegion.etaStrip2);
  etaStripPeak.pk3 = p2eg::getPeakBin20N(ecalRegion.etaStrip3);
  etaStripPeak.pk4 = p2eg::getPeakBin20N(ecalRegion.etaStrip4);
  etaStripPeak.pk5 = p2eg::getPeakBin20N(ecalRegion.etaStrip5);
  etaStripPeak.pk6 = p2eg::getPeakBin20N(ecalRegion.etaStrip6);
  etaStripPeak.pk7 = p2eg::getPeakBin20N(ecalRegion.etaStrip7);
  etaStripPeak.pk8 = p2eg::getPeakBin20N(ecalRegion.etaStrip8);
  etaStripPeak.pk9 = p2eg::getPeakBin20N(ecalRegion.etaStrip9);
  etaStripPeak.pk10 = p2eg::getPeakBin20N(ecalRegion.etaStrip10);
  etaStripPeak.pk11 = p2eg::getPeakBin20N(ecalRegion.etaStrip11);
  etaStripPeak.pk12 = p2eg::getPeakBin20N(ecalRegion.etaStrip12);
  etaStripPeak.pk13 = p2eg::getPeakBin20N(ecalRegion.etaStrip13);
  etaStripPeak.pk14 = p2eg::getPeakBin20N(ecalRegion.etaStrip14);

  p2eg::crystalMax peakIn15;
  peakIn15 = p2eg::getPeakBin15N(etaStripPeak);

  cluster.seedEnergy = peakIn15.energy;
  cluster.energy = 0;
  cluster.etaMax = peakIn15.etaMax;
  cluster.phiMax = peakIn15.phiMax;
  cluster.brems = 0;
  cluster.et5x5 = 0;
  cluster.et2x5 = 0;

  return cluster;
}

//--------------------------------------------------------//

/*
* Return initialized cluster with specified Et, eta, phi, with all other fields (saturation, Et2x5, Et5x5, brems, flags initialized to 0/ false).
*/
p2eg::Cluster p2eg::packCluster(ap_uint<15>& clusterEt, ap_uint<5>& etaMax_t, ap_uint<5>& phiMax_t) {
  ap_uint<12> peggedEt;
  p2eg::Cluster pack;

  ap_uint<5> towerEta = (etaMax_t) / 5;
  ap_uint<2> towerPhi = (phiMax_t) / 5;
  ap_uint<3> clusterEta = etaMax_t - 5 * towerEta;
  ap_uint<3> clusterPhi = phiMax_t - 5 * towerPhi;

  peggedEt = (clusterEt > 0xFFF) ? (ap_uint<12>)0xFFF : (ap_uint<12>)clusterEt;

  pack = p2eg::Cluster(peggedEt, towerEta, towerPhi, clusterEta, clusterPhi, 0);

  return pack;
}

//--------------------------------------------------------//

// Given the cluster seed_eta, seed_phi, and brems, remove the cluster energy
// from the given crystal array temp. Functionally identical to "RemoveTmp".

void p2eg::removeClusterFromCrystal(p2eg::crystal temp[p2eg::CRYSTAL_IN_ETA][p2eg::CRYSTAL_IN_PHI],
                                    ap_uint<5> seed_eta,
                                    ap_uint<5> seed_phi,
                                    ap_uint<2> brems) {
  // Zero out the crystal energies in a 3 (eta) by 5 (phi) window (the clusters are 3x5 in crystals)
  for (int i = 0; i < p2eg::CRYSTAL_IN_ETA; i++) {
    for (int k = 0; k < p2eg::CRYSTAL_IN_PHI; k++) {
      if ((i >= (seed_eta - 1)) && (i <= (seed_eta + 1)) && (k >= (seed_phi - 2)) && (k <= (seed_phi + 2))) {
        temp[i][k].energy = 0;
      }
    }
  }

  // If brems flag is 1, *also* zero the energies in the 3x5 window to the "left" of the cluster
  // N.B. in the positive eta cards, "left" in the region = towards negative phi,
  // but for negative eta cards, everything is flipped, so "left" in the region" = towards positive phi
  if (brems == 1) {
    for (int i = 0; i < p2eg::CRYSTAL_IN_ETA; i++) {
      for (int k = 0; k < p2eg::CRYSTAL_IN_PHI; k++) {
        if ((i >= (seed_eta - 1)) && (i <= (seed_eta + 1)) && (k >= (seed_phi - 2 - 5)) && (k <= (seed_phi + 2 - 5))) {
          temp[i][k].energy = 0;
        }
      }
    }
  }
  // If brems flag is 2, *also* zero the energies in the 3x5 window to the "right" of the cluster
  // N.B. in the positive eta cards, "right" in the region = towards POSITIVE phi,
  // but for negative eta cards, everything is flipped, so "right" in the region = towards NEGATIVE phi
  else if (brems == 2) {
    for (int i = 0; i < p2eg::CRYSTAL_IN_ETA; i++) {
      for (int k = 0; k < p2eg::CRYSTAL_IN_PHI; k++) {
        if ((i >= (seed_eta - 1)) && (i <= (seed_eta + 1)) && (k >= (seed_phi - 2 + 5)) && (k <= (seed_phi + 2 + 5))) {
          temp[i][k].energy = 0;
        }
      }
    }
  }
}

//--------------------------------------------------------//

// Given a 15x20 crystal tempX, and a seed with seed_eta and seed_phi, return a clusterInfo containing
// the cluster energy for a positive bremmstrahulung shift

p2eg::clusterInfo p2eg::getBremsValuesPos(p2eg::crystal tempX[p2eg::CRYSTAL_IN_ETA][p2eg::CRYSTAL_IN_PHI],
                                          ap_uint<5> seed_eta,
                                          ap_uint<5> seed_phi) {
  ap_uint<12> temp[p2eg::CRYSTAL_IN_ETA + 2][p2eg::CRYSTAL_IN_PHI + 4];
  ap_uint<12> phi0eta[3], phi1eta[3], phi2eta[3], phi3eta[3], phi4eta[3];
  ap_uint<12> eta_slice[3];
  p2eg::clusterInfo cluster_tmp;

  // Set all entries in a new ((15+2)x(20+4)) array to be zero.
  for (int i = 0; i < (p2eg::CRYSTAL_IN_ETA + 2); i++) {
    for (int j = 0; j < (p2eg::CRYSTAL_IN_PHI + 4); j++) {
      temp[i][j] = 0;
    }
  }

  // Read the energies of the input crystal tempX into the slightly larger array temp, with an offset so temp is tempX
  // except shifted +1 in eta, and -3 in phi.
  for (int i = 0; i < (p2eg::CRYSTAL_IN_ETA); i++) {
    for (int j = 0; j < (p2eg::CRYSTAL_IN_PHI - 3); j++) {
      temp[i + 1][j] = tempX[i][j + 3].energy;
    }
  }

  ap_uint<6> seed_eta1, seed_phi1;
  seed_eta1 = seed_eta;  //to start from corner
  seed_phi1 = seed_phi;  //to start from corner

  // now we are in the left bottom corner
  // Loop over the shifted array, and at the original location of the seed (seed_eta1/seed_phi1),
  // read a 3 (eta) x 5 (phi) rectangle of crystals where the original location of the seed is in the bottom left corner
  for (int j = 0; j < p2eg::CRYSTAL_IN_ETA; j++) {
    if (j == seed_eta1) {
      for (int k = 0; k < p2eg::CRYSTAL_IN_PHI; k++) {
        if (k == seed_phi1) {
          // Same eta as the seed, read next five crystals in phi
          phi0eta[0] = temp[j][k];
          phi1eta[0] = temp[j][k + 1];
          phi2eta[0] = temp[j][k + 2];
          phi3eta[0] = temp[j][k + 3];
          phi4eta[0] = temp[j][k + 4];

          // +1 eta from the seed, read next five crystals in phi
          phi0eta[1] = temp[j + 1][k];
          phi1eta[1] = temp[j + 1][k + 1];
          phi2eta[1] = temp[j + 1][k + 2];
          phi3eta[1] = temp[j + 1][k + 3];
          phi4eta[1] = temp[j + 1][k + 4];

          // +2 eta from the seed, read next five crystals in phi
          phi0eta[2] = temp[j + 2][k];
          phi1eta[2] = temp[j + 2][k + 1];
          phi2eta[2] = temp[j + 2][k + 2];
          phi3eta[2] = temp[j + 2][k + 3];
          phi4eta[2] = temp[j + 2][k + 4];

          continue;
        }
      }
    }
  }

  // Add up the energies in this 3x5 of crystals, initialize a cluster_tmp, and return it
  for (int i = 0; i < 3; i++) {
    eta_slice[i] = phi0eta[i] + phi1eta[i] + phi2eta[i] + phi3eta[i] + phi4eta[i];
  }
  cluster_tmp.energy = (eta_slice[0] + eta_slice[1] + eta_slice[2]);

  return cluster_tmp;
}

//--------------------------------------------------------//

// Given a 15x20 crystal tempX, and a seed with seed_eta and seed_phi, return a clusterInfo containing
// the cluster energy for a *negative* bremmstrahlung shift

p2eg::clusterInfo p2eg::getBremsValuesNeg(p2eg::crystal tempX[p2eg::CRYSTAL_IN_ETA][p2eg::CRYSTAL_IN_PHI],
                                          ap_uint<5> seed_eta,
                                          ap_uint<5> seed_phi) {
  ap_uint<12> temp[p2eg::CRYSTAL_IN_ETA + 2][p2eg::CRYSTAL_IN_PHI + 4];
  ap_uint<12> phi0eta[3], phi1eta[3], phi2eta[3], phi3eta[3], phi4eta[3];

  ap_uint<12> eta_slice[3];

  p2eg::clusterInfo cluster_tmp;

  // Initialize all entries in a new ((15+2)x(20+4)) array to be zero.
  for (int i = 0; i < (p2eg::CRYSTAL_IN_ETA + 2); i++) {
    for (int j = 0; j < (p2eg::CRYSTAL_IN_PHI + 4); j++) {
      temp[i][j] = 0;
    }
  }

  // Read the energies of the input crystal tempX into the slightly larger array temp, with an offset so temp is tempX
  // except shifted in +1 in eta and +7 in phi
  for (int i = 0; i < (p2eg::CRYSTAL_IN_ETA); i++) {
    for (int j = 0; j < (p2eg::CRYSTAL_IN_PHI - 1); j++) {
      temp[i + 1][j + 7] = tempX[i][j].energy;
    }
  }

  ap_uint<6> seed_eta1, seed_phi1;
  seed_eta1 = seed_eta;  //to start from corner
  seed_phi1 = seed_phi;  //to start from corner

  // Loop over the shifted array, and at the original location of the seed (seed_eta1/seed_phi1),
  // read a 3 (eta) x 5 (phi) rectangle of crystals where the original location of the seed is in the bottom left corner
  for (int j = 0; j < p2eg::CRYSTAL_IN_ETA; j++) {
    if (j == seed_eta1) {
      for (int k = 0; k < p2eg::CRYSTAL_IN_PHI; k++) {
        if (k == seed_phi1) {
          // Same eta as the seed, read next five crystals in phi
          phi0eta[0] = temp[j][k];
          phi1eta[0] = temp[j][k + 1];
          phi2eta[0] = temp[j][k + 2];
          phi3eta[0] = temp[j][k + 3];
          phi4eta[0] = temp[j][k + 4];

          // +1 eta from the seed, read next five crystals in phi
          phi0eta[1] = temp[j + 1][k];
          phi1eta[1] = temp[j + 1][k + 1];
          phi2eta[1] = temp[j + 1][k + 2];
          phi3eta[1] = temp[j + 1][k + 3];
          phi4eta[1] = temp[j + 1][k + 4];

          // +2 eta from the seed, read next five crystals in phi
          phi0eta[2] = temp[j + 2][k];
          phi1eta[2] = temp[j + 2][k + 1];
          phi2eta[2] = temp[j + 2][k + 2];
          phi3eta[2] = temp[j + 2][k + 3];
          phi4eta[2] = temp[j + 2][k + 4];
          continue;
        }
      }
    }
  }

  // Add up the energies in this 3x5 of crystals, initialize a cluster_tmp, and return it
  for (int i = 0; i < 3; i++) {
    eta_slice[i] = phi0eta[i] + phi1eta[i] + phi2eta[i] + phi3eta[i] + phi4eta[i];
  }
  cluster_tmp.energy = (eta_slice[0] + eta_slice[1] + eta_slice[2]);

  return cluster_tmp;
}

//--------------------------------------------------------//

// Given a 15x20 crystal tempX, and a seed with seed_eta and seed_phi, return a clusterInfo containing
// the cluster energy (central value)

p2eg::clusterInfo p2eg::getClusterValues(p2eg::crystal tempX[p2eg::CRYSTAL_IN_ETA][p2eg::CRYSTAL_IN_PHI],
                                         ap_uint<5> seed_eta,
                                         ap_uint<5> seed_phi) {
  ap_uint<12> temp[p2eg::CRYSTAL_IN_ETA + 4][p2eg::CRYSTAL_IN_PHI + 4];
  ap_uint<12> phi0eta[5], phi1eta[5], phi2eta[5], phi3eta[5], phi4eta[5];
  ap_uint<12> eta_slice[5];
  ap_uint<12> et2x5_1Tot, et2x5_2Tot, etSum2x5;
  ap_uint<12> et5x5Tot;

  p2eg::clusterInfo cluster_tmp;
  // Initialize empty (15+4)x(20+4) array
  for (int i = 0; i < (p2eg::CRYSTAL_IN_ETA + 4); i++) {
    for (int k = 0; k < (p2eg::CRYSTAL_IN_PHI + 4); k++) {
      temp[i][k] = 0;
    }
  }

  // Copy input array energies into temp array with +2 eta and +2 phi offset.
  for (int i = 0; i < (p2eg::CRYSTAL_IN_ETA); i++) {
    for (int k = 0; k < (p2eg::CRYSTAL_IN_PHI); k++) {
      temp[i + 2][k + 2] = tempX[i][k].energy;
    }
  }

  ap_uint<6> seed_eta1, seed_phi1;
  seed_eta1 = seed_eta;  //to start from corner
  seed_phi1 = seed_phi;  //to start from corner

  // now we are in the left bottom corner
  // Loop over the shifted array, and at the original location of the seed (seed_eta1/seed_phi1),
  // read a 5 (eta) x 5 (phi) rectangle of crystals where the original location of the seed is in the bottom left corner
  for (int j = 0; j < p2eg::CRYSTAL_IN_ETA; j++) {
    if (j == seed_eta1) {
      for (int k = 0; k < p2eg::CRYSTAL_IN_PHI; k++) {
        if (k == seed_phi1) {
          // Same eta as the seed, read next five crystals in phi
          phi0eta[0] = temp[j][k];
          phi1eta[0] = temp[j][k + 1];
          phi2eta[0] = temp[j][k + 2];
          phi3eta[0] = temp[j][k + 3];
          phi4eta[0] = temp[j][k + 4];

          // +1 eta from the seed, read next five crystals in phi
          phi0eta[1] = temp[j + 1][k];
          phi1eta[1] = temp[j + 1][k + 1];
          phi2eta[1] = temp[j + 1][k + 2];
          phi3eta[1] = temp[j + 1][k + 3];
          phi4eta[1] = temp[j + 1][k + 4];

          // +2 eta from the seed, read next five crystals in phi
          phi0eta[2] = temp[j + 2][k];
          phi1eta[2] = temp[j + 2][k + 1];
          phi2eta[2] = temp[j + 2][k + 2];
          phi3eta[2] = temp[j + 2][k + 3];
          phi4eta[2] = temp[j + 2][k + 4];

          // +3 eta from the seed, read next five crystals in phi
          phi0eta[3] = temp[j + 3][k];
          phi1eta[3] = temp[j + 3][k + 1];
          phi2eta[3] = temp[j + 3][k + 2];
          phi3eta[3] = temp[j + 3][k + 3];
          phi4eta[3] = temp[j + 3][k + 4];

          // +4 eta from the seed, read next five crystals in phi
          phi0eta[4] = temp[j + 4][k];
          phi1eta[4] = temp[j + 4][k + 1];
          phi2eta[4] = temp[j + 4][k + 2];
          phi3eta[4] = temp[j + 4][k + 3];
          phi4eta[4] = temp[j + 4][k + 4];

          continue;
        }
      }
    }
  }

  // Add the first three eta strips into the cluster energy
  for (int i = 0; i < 5; i++) {
    eta_slice[i] = phi0eta[i] + phi1eta[i] + phi2eta[i] + phi3eta[i] + phi4eta[i];
  }

  cluster_tmp.energy = (eta_slice[1] + eta_slice[2] + eta_slice[3]);

  // Get the energy totals in the 5x5 and also in two 2x5
  et5x5Tot = (eta_slice[0] + eta_slice[1] + eta_slice[2] + eta_slice[3] + eta_slice[4]);
  et2x5_1Tot = (eta_slice[1] + eta_slice[2]);
  et2x5_2Tot = (eta_slice[2] + eta_slice[3]);

  if (et2x5_1Tot >= et2x5_2Tot)
    etSum2x5 = et2x5_1Tot;
  else
    etSum2x5 = et2x5_2Tot;

  cluster_tmp.et5x5 = et5x5Tot;
  cluster_tmp.et2x5 = etSum2x5;

  return cluster_tmp;
}

//--------------------------------------------------------//

// In 15x20 crystal array temp, return the next cluster, and remove the cluster's energy
// from the crystal array.

p2eg::Cluster p2eg::getClusterFromRegion3x4(p2eg::crystal temp[p2eg::CRYSTAL_IN_ETA][p2eg::CRYSTAL_IN_PHI]) {
  p2eg::Cluster returnCluster;
  p2eg::clusterInfo cluster_tmp;
  p2eg::clusterInfo cluster_tmpCenter;
  p2eg::clusterInfo cluster_tmpBneg;
  p2eg::clusterInfo cluster_tmpBpos;

  p2eg::ecalRegion_t ecalRegion;
  ecalRegion = p2eg::initStructure(temp);

  cluster_tmp = p2eg::getClusterPosition(ecalRegion);

  float seedEnergyFloat = cluster_tmp.seedEnergy / 8.0;

  // Do not make cluster if seed is less than 1.0 GeV
  if (seedEnergyFloat < 1.0) {
    cluster_tmp.energy = 0;
    cluster_tmp.phiMax = 0;
    cluster_tmp.etaMax = 0;
    return p2eg::packCluster(cluster_tmp.energy, cluster_tmp.phiMax, cluster_tmp.etaMax);
  }

  ap_uint<5> seed_phi = cluster_tmp.phiMax;
  ap_uint<5> seed_eta = cluster_tmp.etaMax;

  cluster_tmpCenter = p2eg::getClusterValues(temp, seed_eta, seed_phi);
  cluster_tmpBneg = p2eg::getBremsValuesNeg(temp, seed_eta, seed_phi);
  cluster_tmpBpos = p2eg::getBremsValuesPos(temp, seed_eta, seed_phi);

  cluster_tmp.energy = cluster_tmpCenter.energy;
  cluster_tmp.brems = 0;

  // Create a cluster
  if ((cluster_tmpBneg.energy > cluster_tmpCenter.energy / 8) && (cluster_tmpBneg.energy > cluster_tmpBpos.energy)) {
    cluster_tmp.energy = (cluster_tmpCenter.energy + cluster_tmpBneg.energy);
    cluster_tmp.brems = 1;
  } else if (cluster_tmpBpos.energy > cluster_tmpCenter.energy / 8) {
    cluster_tmp.energy = (cluster_tmpCenter.energy + cluster_tmpBpos.energy);
    cluster_tmp.brems = 2;
  }

  returnCluster = p2eg::packCluster(cluster_tmp.energy, cluster_tmp.etaMax, cluster_tmp.phiMax);
  p2eg::removeClusterFromCrystal(temp, seed_eta, seed_phi, cluster_tmp.brems);

  // Add clusterInfo members to the output cluster members
  returnCluster.brems = cluster_tmp.brems;
  returnCluster.et5x5 = cluster_tmpCenter.et5x5;  // get et5x5 from the center value
  returnCluster.et2x5 = cluster_tmpCenter.et2x5;  // get et2x5 from the center value

  return returnCluster;
}

//--------------------------------------------------------//

// Stitch clusters in cluster_list across the boundary specified by
// towerEtaUpper and towerEtaLower (using RCT card notation). Modifies the input vector
// (passed by reference). If two clusters are combined, modify the higher-energy cluster and
// zero out the energy of the smaller-energy cluster.
// cc is the RCT card number (for print-out statements only).

void p2eg::stitchClusterOverRegionBoundary(std::vector<Cluster>& cluster_list,
                                           int towerEtaUpper,
                                           int towerEtaLower,
                                           int cc) {
  (void)cc;  // for printout statements

  int crystalEtaUpper = 0;
  int crystalEtaLower = 4;

  for (size_t i = 0; i < cluster_list.size(); i++) {
    for (size_t j = 0; j < cluster_list.size(); j++) {
      // Do not double-count
      if (i == j)
        continue;

      p2eg::Cluster c1 = cluster_list[i];
      p2eg::Cluster c2 = cluster_list[j];

      p2eg::Cluster newc1;
      p2eg::Cluster newc2;

      // Use the .towerEtaInCard() method to get the tower eta in the entire RCT card
      if ((c1.clusterEnergy() > 0) && (c1.towerEtaInCard() == towerEtaUpper) && (c1.clusterEta() == crystalEtaUpper)) {
        if ((c2.clusterEnergy() > 0) && (c2.towerEtaInCard() == towerEtaLower) &&
            (c2.clusterEta() == crystalEtaLower)) {
          ap_uint<5> phi1 = c1.towerPhi() * 5 + c1.clusterPhi();
          ap_uint<5> phi2 = c2.towerPhi() * 5 + c2.clusterPhi();
          ap_uint<5> dPhi;
          dPhi = (phi1 > phi2) ? (phi1 - phi2) : (phi2 - phi1);

          if (dPhi < 2) {
            ap_uint<15> totalEnergy = c1.clusterEnergy() + c2.clusterEnergy();
            ap_uint<15> totalEt2x5 = c1.uint_et2x5() + c2.uint_et2x5();
            ap_uint<15> totalEt5x5 = c1.uint_et5x5() + c2.uint_et5x5();

            bool rct_is_iso = false;         // RCT has no isolation information
            bool rct_is_looseTkiso = false;  // RCT has no isolation information

            // Initialize a cluster with the larger cluster's position and total energy
            if (c1.clusterEnergy() > c2.clusterEnergy()) {
              newc1 = p2eg::Cluster(totalEnergy,
                                    c1.towerEta(),
                                    c1.towerPhi(),
                                    c1.clusterEta(),
                                    c1.clusterPhi(),
                                    c1.satur(),
                                    totalEt5x5,
                                    totalEt2x5,
                                    c1.getBrems(),
                                    c1.getIsSS(),
                                    c1.getIsLooseTkss(),
                                    rct_is_iso,
                                    rct_is_looseTkiso,
                                    c1.region());
              newc2 = p2eg::Cluster(0,
                                    c2.towerEta(),
                                    c2.towerPhi(),
                                    c2.clusterEta(),
                                    c2.clusterPhi(),
                                    c2.satur(),
                                    0,
                                    0,
                                    0,
                                    false,
                                    false,
                                    rct_is_iso,
                                    rct_is_looseTkiso,
                                    c2.region());
              cluster_list[i] = newc1;
              cluster_list[j] = newc2;
            } else {
              // Analogous to above portion
              newc1 = p2eg::Cluster(0,
                                    c1.towerEta(),
                                    c1.towerPhi(),
                                    c1.clusterEta(),
                                    c1.clusterPhi(),
                                    c1.satur(),
                                    0,
                                    0,
                                    0,
                                    false,
                                    false,
                                    rct_is_iso,
                                    rct_is_looseTkiso,
                                    c1.region());
              newc2 = p2eg::Cluster(totalEnergy,
                                    c2.towerEta(),
                                    c2.towerPhi(),
                                    c2.clusterEta(),
                                    c2.clusterPhi(),
                                    c2.satur(),
                                    totalEt5x5,
                                    totalEt2x5,
                                    c2.getBrems(),
                                    c2.getIsSS(),
                                    c2.getIsLooseTkss(),
                                    rct_is_iso,
                                    rct_is_looseTkiso,
                                    c2.region());
              cluster_list[i] = newc1;
              cluster_list[j] = newc2;
            }
          }
        }
      }
    }
  }
}

//--------------------------------------------------------//

#endif

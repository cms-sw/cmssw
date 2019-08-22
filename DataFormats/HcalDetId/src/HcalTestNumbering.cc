///////////////////////////////////////////////////////////////////////////////
// File: HcalTestNumbering.cc
// Description: Numbering scheme packing for test beam hadron calorimeter
///////////////////////////////////////////////////////////////////////////////
#include "DataFormats/HcalDetId/interface/HcalTestNumbering.h"

uint32_t HcalTestNumbering::packHcalIndex(int det, int z, int depth, int eta, int phi, int lay) {
  uint32_t idx = (det & 15) << 28;  //bits 28-31
  idx += ((depth - 1) & 3) << 26;   //bits 26-27
  idx += ((lay - 1) & 31) << 21;    //bits 21-25
  idx += (z & 1) << 20;             //bits 20
  idx += (eta & 1023) << 10;        //bits 10-19
  idx += (phi & 1023);              //bits  0-9

  return idx;
}

void HcalTestNumbering::unpackHcalIndex(
    const uint32_t& idx, int& det, int& z, int& depth, int& eta, int& phi, int& lay) {
  det = (idx >> 28) & 15;
  depth = (idx >> 26) & 3;
  depth += 1;
  lay = (idx >> 21) & 31;
  lay += 1;
  z = (idx >> 20) & 1;
  eta = (idx >> 10) & 1023;
  phi = (idx & 1023);
}

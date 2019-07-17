/*
 * MuCorrelatorConfig.cc
 *
 *  Created on: Jan 30, 2019
 *      Author: Karol Bunkowski kbunkow@cern.ch
 */

#include "L1Trigger/L1TMuonBayes/interface/MuCorrelator/MuCorrelatorConfig.h"

#include <math.h>
#include <limits>
#include <sstream>
#include <iomanip>

MuCorrelatorConfig::MuCorrelatorConfig() {
  buildPtHwBins();
}

//TODO should be read from config
void MuCorrelatorConfig::buildPtHwBins() {
  float stride = 1;
  int iPt = 6;
  for(unsigned int ptBin = 0; ptBin < ptBins; ptBin++) {
    ptHwBins.push_back(iPt);

    if(ptBin < 8)
      stride = 1;
    else if(ptBin < 12)
      stride = 1;
    else if(ptBin < 24)
      stride = 2;
    else if(ptBin < 32)
      stride = 3;
    else if(ptBin < 40)
      stride = 4;
    else if(ptBin < 48)
      stride = 6;
    else if(ptBin < 56)
      stride += 4;
    else
      stride += 8;

    iPt += stride;
  }
  ptHwBins.push_back(std::numeric_limits<int>::max());
}

unsigned int MuCorrelatorConfig::logLayerToRefLayar(unsigned int logicLayer, unsigned int etaBin) const {
  //TODO implement
  return 0;
}

int MuCorrelatorConfig::getProcScalePhi(double phiRad, double procPhiZeroRad) const {
  const double phiUnit = 2*M_PI/nPhiBins(); //rad/unit

  // local angle in CSC halfStrip usnits
  return foldPhi( lround ( (phiRad - procPhiZeroRad)/phiUnit ) );
}

float MuCorrelatorConfig::getProcScalePhiToRad(int phiHw) const {
  const double phiUnit = 2*M_PI/nPhiBins();
  return phiHw * phiUnit;
}

unsigned int MuCorrelatorConfig::ptGeVToPtBin(float ptGeV) const {
  //TODO implement nonlinear scale;
  //ptBin = ptHw / 8; //TODO some implementation, probably not optimal, do it in a batter way
/*  double  ptBin = 40 * log10(1 + (ptGeV - 2.5) * 0.15);
  if(ptBin < 0)
    return 0;
  if(ptBin >= ptBins)
    return ptBins -1;

  return  round(ptBin);*/

  return ptHwToPtBin(ptGevToHw(ptGeV));
}

unsigned int MuCorrelatorConfig::ptHwToPtBin(int ptHw) const {
  for(unsigned int ptBin = 0; ptBin < ptBins; ptBin++) {
    if(ptHwBins[ptBin] >= ptHw)
      return ptBin;
  }

  return ptBins -1; //"to inf" bin
}

unsigned int MuCorrelatorConfig::etaHwToEtaBin(int etaHw) const {
  int endcapBorder = 80; //= 0.87
  if(abs(etaHw) < endcapBorder)
    return 0;

  //TODO optimize e.g. use scale 8 (easier for firmware), but then check what is the eta dependence above eta 2.175
  //if 10 is kept then use firmware friendly division, i.e. e.g.:
  //((abs(etaHw) - endcapBorder) * 102) >> 10 where 102 = 2^10 / 10
  int scale = 10;
  unsigned int etaBin = (abs(etaHw) - endcapBorder)/scale;

  if(etaBin < etaBins)
    return etaBin;

  return etaBins-1;

}
/*
 0  - MB1 phi
 1  - MB1 phiB
 2  - MB2 phi
 3  - MB3 phiB
 4  - MB3 phi
 5  - MB3 phiB
 6  - MB4 phi
 7  - MB5 phiB

 8  - ME1/1 phi
 9    - ME1/2 and ME1/3 phi
 10 - ME2 phi
 11 - ME3 phi
 12 - ME4 phi

 13 - RB1in phi
 14 - RB1out phi
 15 - RB2in phi
 16 - RB2out phi
 17 - RB3 phi
 18 - RB4 phi

 19 - RE1 phi
 20 - RE2 phi
 21 - RE3 phi
 22 - RE4 phi

 23 - MB1 eta
 24 - MB2 eta
 25 - MB3 eta
    - MB4 has no eta TODO check
//TODO add separate eta layer for the ME1/1???
 26 - ME1 eta
 27 - ME2 eta
 28 - ME3 eta
 29 - ME4 eta
 */
bool MuCorrelatorConfig::isEndcapLayer(unsigned int layer) const {
  if(layer >= 8 && layer <= 12)
    return true;

  if(layer >= 19 && layer <= 22)
    return true;

  if(layer >= 26 && layer <= 29)
    return true;

  return false;
}

bool MuCorrelatorConfig::isPhiLayer(unsigned int layer) const {
  if(layer < phiLayers)
    return true;
  return false;
}


bool MuCorrelatorConfig::isBendingLayer(unsigned int layer) const {
  if(layer == 1 || layer == 3 || layer == 5 || layer == 7) {
    return true;
  }
  return false;
}

std::string MuCorrelatorConfig::ptBinString(unsigned int ptBin, int mode) const {
  int ptHwLow = ptBin > 0 ? ptHwBins.at(ptBin - 1) : 0;
  int ptHwUp = ptHwBins.at(ptBin);
  std::ostringstream ostr;
  if(mode == 0)
    ostr<<"ptHw: "<<std::setw(3)<<ptHwLow+1<<" - "<<std::setw(3)<<ptHwUp;
  else if(mode == 1)
    ostr<<hwPtToGev(ptHwLow+1)<<" - "<<hwPtToGev(ptHwUp+1)<<" GeV";

  return ostr.str();
}

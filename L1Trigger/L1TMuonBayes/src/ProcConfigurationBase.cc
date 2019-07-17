/*
 * ProcConfigurationBase.cc
 *
 *  Created on: Jan 30, 2019
 *      Author: Karol Bunkowski kbunkow@cern.ch
 */

#include "L1Trigger/L1TMuonBayes/interface/ProcConfigurationBase.h"

ProcConfigurationBase::ProcConfigurationBase() {

}

ProcConfigurationBase::~ProcConfigurationBase() {

}


int ProcConfigurationBase::foldPhi(int phi) const {
  int phiBins = nPhiBins();
  if(phi > phiBins/2)
    return (phi - phiBins );
  else if(phi < -phiBins /2)
    return (phi + phiBins );

  return phi;
}

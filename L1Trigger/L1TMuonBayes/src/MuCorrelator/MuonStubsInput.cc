/*
 * MuonStubsInput.cc
 *
 *  Created on: Jan 31, 2019
 *      Author: Karol Bunkowski kbunkow@cern.ch
 */

#include "L1Trigger/L1TMuonBayes/interface/MuCorrelator/MuonStubsInput.h"

MuonStubsInput::MuonStubsInput(MuCorrelatorConfigPtr& config): config(config), muonStubsInLayers(config->nLayers()) {

}


std::ostream & operator<< (std::ostream &out, const MuonStubsInput& stubsInput) {
  out <<"MuonStubsInput: "<<std::endl;
  for(auto& layerStubs : stubsInput.getMuonStubs()) {
    for(auto& stub : layerStubs) {
      out <<(*stub)<<std::endl;
    }
  }
  return out;
}




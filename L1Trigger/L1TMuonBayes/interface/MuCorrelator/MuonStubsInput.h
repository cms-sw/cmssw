/*
 * MuonStubsInput.h
 *
 *  Created on: Jan 31, 2019
 *      Author: Karol Bunkowski kbunkow@cern.ch
 */

#ifndef MUCORRELATOR_MUONSTUBSINPUT_H_
#define MUCORRELATOR_MUONSTUBSINPUT_H_

#include "L1Trigger/L1TMuonBayes/interface/MuCorrelator/MuCorrelatorConfig.h"
#include "L1Trigger/L1TMuonBayes/interface/MuonStub.h"
#include <ostream>

class MuonStubsInput {
public:
  MuonStubsInput(MuCorrelatorConfigPtr& config);

  //void addStub(unsigned int iLayer, unsigned int iInput, MuonStubPtr& stub);
  void addStub(unsigned int iLayer, const MuonStubPtr& stub) {
    muonStubsInLayers.at(iLayer).emplace_back(stub);
  }

  ///Reset vectors with data.
  //void clear();

  ///Read data from a XML file
/*  void readData(XMLConfigReader *aReader,
      unsigned int iEvent=0,
      unsigned int iProcessor=0);*/


  //MuonStub getMuonStub(unsigned int iLayer, unsigned int iInput) const;

  MuonStubPtrs2D& getMuonStubs() {
    return muonStubsInLayers;
  }

  const MuonStubPtrs2D& getMuonStubs() const {
    return muonStubsInLayers;
  }

  friend std::ostream & operator << (std::ostream &out, const MuonStubsInput& stubsInput);

private:
  MuCorrelatorConfigPtr config;

  //[iLayer][iStub]
  MuonStubPtrs2D muonStubsInLayers;
};


#endif /* INTERFACE_MUCORRELATOR_MUONSTUBSINPUT_H_ */

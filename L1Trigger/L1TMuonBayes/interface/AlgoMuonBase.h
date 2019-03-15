/*
 * AlgoMuonBase.h
 *
 *  Created on: Mar 1, 2019
 *      Author: Karol Bunkowski kbunkow@cern.ch
 */

#ifndef INTERFACE_ALGOMUONBASE_H_
#define INTERFACE_ALGOMUONBASE_H_

#include "L1Trigger/L1TMuonBayes/interface/MuonStub.h"
#include "L1Trigger/L1TMuonBayes/interface/StubResult.h"

class AlgoMuonBase {
public:
  AlgoMuonBase();
  virtual ~AlgoMuonBase();

  virtual int getEtaHw() const = 0;

  virtual bool isValid() const = 0;

  //virtual void setValid(bool valid) = 0;

  virtual unsigned int getFiredLayerCnt() const = 0;

  virtual double getPdfSum() const = 0;

 /* virtual const bool isKilled() const = 0;

  virtual void kill()  = 0;*/

  //virtual bool isLayerFired(unsigned int iLayer) const  = 0;

  virtual const StubResult& getStubResult(unsigned int iLayer) const  = 0;

  virtual const StubResults& getStubResults() const  = 0;

  //virtual const boost::dynamic_bitset<>& getFiredLayerBits() const  = 0;


  virtual void setBeta(float beta) {
    this->beta = beta;
  }

  virtual float getBeta() const {
    return beta;
  }

  //index in the tTTrackHandle or in the SimTrackHanlde, needed for generation of patterns etc. not for firmware
  virtual unsigned int getTrackIndex() const = 0;

  virtual double getSimBeta() const {
    return 0;
  }

private:
  float beta = 0; //zero means it is not measured
};

#endif /* INTERFACE_ALGOMUONBASE_H_ */

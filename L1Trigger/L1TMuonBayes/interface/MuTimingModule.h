/*
 * MuTimingModule.h
 *
 *  Created on: Mar 7, 2019
 *      Author: Karol Bunkowski kbunkow@cern.ch
 */

#ifndef INTERFACE_MUTIMINGMODULE_H_
#define INTERFACE_MUTIMINGMODULE_H_

#include "L1Trigger/L1TMuonBayes/interface/AlgoMuonBase.h"
#include "L1Trigger/L1TMuonBayes/interface/ProcConfigurationBase.h"

#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>

class MuTimingModule {
public:
  MuTimingModule(const ProcConfigurationBase* config);

  virtual ~MuTimingModule();

  virtual void process(AlgoMuonBase* algoMuon);

  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
      ar & BOOST_SERIALIZATION_NVP(rolls);
      ar & BOOST_SERIALIZATION_NVP(etaBins);
      ar & BOOST_SERIALIZATION_NVP(timingBins);
      ar & BOOST_SERIALIZATION_NVP(betaBins);

      ar & BOOST_SERIALIZATION_NVP(timigTo1_Beta);
  }

protected:
  const ProcConfigurationBase* config; //TODO is this needed?

  virtual unsigned int etaHwToEtaBin(int trackEtaHw, const MuonStubPtr& muonStub) const;

  virtual unsigned int betaTo1_betaBin(double beta) const;

  virtual float one_betaBinToBeta(unsigned int one_betaBin) const; //one_betaBin one over beta

  virtual unsigned int timingToTimingBin(int timing) const;

  //[layer][roll][etaBin][timing]
  std::vector<std::vector<std::vector<std::vector<int> > > >  timigTo1_Beta; //gives average 1/beta

  //[layer][etaBin][timing]
  //std::vector<std::vector<std::vector<int> > >  timigTo1_Beta;

  unsigned int rolls = 9; //in the barrel abs(wheel) is used
  unsigned int etaBins = 4;
  unsigned int timingBins = 40;
  unsigned int betaBins = 16; //8 per BX, betaBins[0] reserved for no-beta, betaBins[1] - beta = 1
};

#endif /* INTERFACE_MUTIMINGMODULE_H_ */

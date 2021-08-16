/*
 * GpResultsToPt.h
 *
 *  Created on: Mar 6, 2020
 *      Author: kbunkow
 */

#ifndef INTERFACE_OMTF_GPRESULTSTOPT_H_
#define INTERFACE_OMTF_GPRESULTSTOPT_H_

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GoldenPattern.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/AlgoMuon.h"
#include "TH1I.h"

#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>

class GpResultsToPt {
public:
  GpResultsToPt(const std::vector<std::shared_ptr<GoldenPattern> >& gps,
                const OMTFConfiguration* omtfConfig,
                unsigned int lutSize);  //for training

  GpResultsToPt(const std::vector<std::shared_ptr<GoldenPattern> >& gps,
                const OMTFConfiguration* omtfConfig);  //for running, gpResultsToPtLuts should be read from archive
  virtual ~GpResultsToPt();

  unsigned int lutAddres(AlgoMuons::value_type& algoMuon, unsigned int& candProcIndx);

  //return ptCode
  virtual int getValue(AlgoMuons::value_type& algoMuon, unsigned int& candProcIndx);

  virtual void updateStat(AlgoMuons::value_type& algoMuon, unsigned int& candProcIndx, double ptSim, int chargeSim);

  void caluateLutValues();

  const std::vector<std::vector<double> >& getGpResultsToPtLuts() const { return gpResultsStatLuts; }

  void setGpResultsToPtLuts(const std::vector<std::vector<double> >& gpResultsToPtLuts) {
    this->gpResultsStatLuts = gpResultsToPtLuts;
  }

  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar& gpResultsToPtLuts;
  }

private:
  unsigned int lutSize = 1024;
  const std::vector<std::shared_ptr<GoldenPattern> >& gps;

  const OMTFConfiguration* omtfConfig = nullptr;

  std::vector<std::vector<int> > gpResultsToPtLuts;

  std::vector<std::vector<double> > gpResultsStatLuts;  //[iGP][lutAddr] gpResultsStatLuts
  std::vector<std::vector<int> > entries;               //[iGP][lutAddr]

  std::vector<GoldenPattern*> lowerGps;  //[iGP] - the pointer to the pattern with one step higher pt
  std::vector<GoldenPattern*> higerGps;  //[iGP] - the pointer to the pattern with one step lower pt

  std::vector<TH1*> ptGenInPats;
};

#endif /* INTERFACE_OMTF_GPRESULTSTOPT_H_ */

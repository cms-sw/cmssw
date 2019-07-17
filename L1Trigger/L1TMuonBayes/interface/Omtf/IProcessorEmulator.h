/*
 * IProcessor.h
 *
 *  Created on: Oct 4, 2017
 *      Author: kbunkow
 */

#ifndef OMTF_IPROCESSOREMULATOR_H_
#define OMTF_IPROCESSOREMULATOR_H_


#include <L1Trigger/L1TMuonBayes/interface/Omtf/GhostBuster.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/GoldenPatternResult.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/IOMTFEmulationObserver.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/OMTFinput.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/OMTFSorter.h>
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"


class IProcessorEmulator {
public:
  virtual ~IProcessorEmulator() {}

  ///Fill GP vec with patterns from CondFormats object
  //virtual bool configure(const OMTFConfiguration * omtfParams, const L1TMuonOverlapParams* omtfPatterns) = 0;

  virtual const void processInput(unsigned int iProcessor, l1t::tftype mtfType,
                    const OMTFinput & aInput) = 0;

  ///allows to use other sorter implementation than the default one
  //virtual void setSorter(SorterBase<GoldenPatternType>* sorter);

  ///allows to use other IGhostBuster implementation than the default one
  virtual void setGhostBuster(IGhostBuster* ghostBuster) = 0;

  virtual AlgoMuons sortResults(unsigned int iProcessor, l1t::tftype mtfType, int charge = 0) = 0;

  virtual AlgoMuons ghostBust(AlgoMuons refHitCands, int charge=0) = 0;

  virtual bool checkHitPatternValidity(unsigned int hits) = 0;

  virtual std::vector<l1t::RegionalMuonCand> getFinalcandidates(unsigned int iProcessor, l1t::tftype mtfType, const AlgoMuons& algoCands) = 0;


  virtual void loadAndFilterDigis(const edm::Event& iEvent, const edm::ParameterSet& edmCfg) = 0;

  virtual std::vector<l1t::RegionalMuonCand> run(unsigned int iProcessor, l1t::tftype mtfType, int bx, std::vector<std::unique_ptr<IOMTFEmulationObserver> >& observers) = 0;
};



#endif /* OMTF_IPROCESSOREMULATOR_H_ */

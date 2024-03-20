/*
 * IProcessor.h
 *
 *  Created on: Oct 4, 2017
 *      Author: kbunkow
 */

#ifndef L1T_OmtfP1_IPROCESSOREMULATOR_H_
#define L1T_OmtfP1_IPROCESSOREMULATOR_H_

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GhostBuster.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GoldenPatternResult.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/IOMTFEmulationObserver.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFinput.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFSorter.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFinputMaker.h"

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"

class IProcessorEmulator {
public:
  virtual ~IProcessorEmulator() {}

  virtual void processInput(unsigned int iProcessor,
                            l1t::tftype mtfType,
                            const OMTFinput& aInput,
                            std::vector<std::unique_ptr<IOMTFEmulationObserver> >& observers) = 0;

  ///allows to use other IGhostBuster implementation than the default one
  virtual void setGhostBuster(IGhostBuster* ghostBuster) = 0;

  virtual AlgoMuons sortResults(unsigned int iProcessor, l1t::tftype mtfType, int charge = 0) = 0;

  virtual AlgoMuons ghostBust(AlgoMuons refHitCands, int charge = 0) = 0;

  virtual bool checkHitPatternValidity(unsigned int hits) = 0;

  virtual std::vector<l1t::RegionalMuonCand> getFinalcandidates(unsigned int iProcessor,
                                                                l1t::tftype mtfType,
                                                                const AlgoMuons& algoCands) = 0;

  virtual std::vector<l1t::RegionalMuonCand> run(unsigned int iProcessor,
                                                 l1t::tftype mtfType,
                                                 int bx,
                                                 OMTFinputMaker* inputMaker,
                                                 std::vector<std::unique_ptr<IOMTFEmulationObserver> >& observers) = 0;

  virtual void printInfo() const = 0;
};

#endif /* L1T_OmtfP1_IPROCESSOREMULATOR_H_ */

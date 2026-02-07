/*
 * OmtfProcessorPhase2.h
 *
 *  Created on: Oct 29, 2025
 *      Author: kbunkow
 */

#ifndef L1Trigger_L1TMuonOverlapPhase2_OMTFPROCESSORPHASE2_H_
#define L1Trigger_L1TMuonOverlapPhase2_OMTFPROCESSORPHASE2_H_

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFProcessor.h"
#include "L1Trigger/L1TMuonOverlapPhase2/interface/MlModelBase.h"

#include "DataFormats/L1TMuonPhase2/interface/SAMuon.h"

class OmtfProcessorPhase2 {
public:
  //Composition is used over inheritance here, primarily because the OMTFProcessor is a template class and is constructed in OMTFReconstruction
  OmtfProcessorPhase2(const OMTFConfiguration* omtfConfig, const unique_ptr<IProcessorEmulator>& omtfProc);

  virtual ~OmtfProcessorPhase2();

  void beginRun(const edm::ParameterSet& edmParameterSet, edm::EventSetup const& iSetup);

  FinalMuons run(unsigned int iProcessor,
                 l1t::tftype mtfType,
                 int bx,
                 OMTFinputMaker* inputMaker,
                 std::vector<std::unique_ptr<IOMTFEmulationObserver> >& observers);

  void assignQualityPhase2(AlgoMuons::value_type& algoMuon);

  void convertToGmtScalesPhase2(unsigned int iProcessor, l1t::tftype mtfType, FinalMuonPtr& finalMuon);

  l1t::SAMuonCollection getSAMuons(unsigned int iProcessor,
                                   l1t::tftype mtfType,
                                   FinalMuons& finalMuons,
                                   bool costrainedPt);

private:
  const OMTFConfiguration* omtfConfig;

  //reference to the unique_ptr is used here, because the omtfProc might be re-constructed in the OMTFReconstruction each run
  const unique_ptr<IProcessorEmulator>& omtfProc;

  unique_ptr<MlModelBase> mlModel;

  std::map<unsigned int, int> firedLayersToQuality;
};

#endif /* L1Trigger_L1TMuonOverlapPhase2_OMTFPROCESSORPHASE2_H_ */

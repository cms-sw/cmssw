#ifndef OMTF_OMTFProcessor_H
#define OMTF_OMTFProcessor_H

#include <L1Trigger/L1TMuonBayes/interface/Omtf/AlgoMuon.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/GoldenPattern.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/GoldenPatternResult.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/IGhostBuster.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/IProcessorEmulator.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/OMTFConfiguration.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/OMTFinputMaker.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/ProcessorBase.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/SorterBase.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/SorterBase.h>
#include <memory>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

#include "L1Trigger/L1TMuonBayes/interface/Omtf/GoldenPattern.h"


class OMTFinput;

namespace edm{
class ParameterSet;
};

template <class GoldenPatternType>
class OMTFProcessor: public ProcessorBase<GoldenPatternType>, public IProcessorEmulator {
 public:

  OMTFProcessor(OMTFConfiguration* omtfConfig, const edm::ParameterSet& edmCfg, edm::EventSetup const& evSetup, const L1TMuonOverlapParams* omtfPatterns, MuStubsInputTokens& muStubsInputTokens);

  OMTFProcessor(OMTFConfiguration* omtfConfig, const edm::ParameterSet& edmCfg, edm::EventSetup const& evSetup, const typename ProcessorBase<GoldenPatternType>::GoldenPatternVec& gps, MuStubsInputTokens& muStubsInputTokens);

  virtual ~OMTFProcessor();

  ///Fill GP vec with patterns from CondFormats object
/*  virtual bool configure(const OMTFConfiguration* omtfParams, const L1TMuonOverlapParams* omtfPatterns) {
    return ProcessorBase<GoldenPatternType>::configure(omtfParams, omtfPatterns);
  }*/

   ///Process input data from a single event
  ///Input data is represented by hits in logic layers expressed in local coordinates
  ///Vector index: number of the ref hit (from 0 to nTestRefHits i.e. 4)
  ///Map key: GoldenPattern key
  //const std::vector<OMTFProcessor::resultsMap> &
  virtual const void processInput(unsigned int iProcessor, l1t::tftype mtfType,
							      const OMTFinput & aInput);
  
  virtual AlgoMuons sortResults(unsigned int iProcessor, l1t::tftype mtfType, int charge=0);

  virtual AlgoMuons ghostBust(AlgoMuons refHitCands, int charge=0) {
    return ghostBuster->select(refHitCands, charge);
  }

  //convert algo muon to outgoing Candidates
  virtual std::vector<l1t::RegionalMuonCand> getFinalcandidates(
                 unsigned int iProcessor, l1t::tftype mtfType,
                 const AlgoMuons& algoCands);

  ///allows to use other sorter implementation than the default one
  virtual void setSorter(SorterBase<GoldenPatternType>* sorter) {
    this->sorter.reset(sorter);
  }

  ///allows to use other IGhostBuster implementation than the default one
  virtual void setGhostBuster(IGhostBuster* ghostBuster) {
    this->ghostBuster.reset(ghostBuster);
  }

  virtual void loadAndFilterDigis(const edm::Event& iEvent, const edm::ParameterSet& edmCfg);

  virtual std::vector<l1t::RegionalMuonCand> run(unsigned int iProcessor, l1t::tftype mtfType, int bx, std::vector<std::unique_ptr<IOMTFEmulationObserver> >& observers);

protected:
  OMTFinputMaker       inputMaker;

 private:
  virtual void init(const edm::ParameterSet& edmCfg, edm::EventSetup const& evSetup, MuStubsInputTokens& muStubsInputTokens);

  ///Check if the hit pattern of given OMTF candite is not on the list
  ///of invalid hit patterns. Invalid hit patterns provode very little
  ///to efficiency, but gives high contribution to rate.
  ///Candidate with invalid hit patterns is assigned quality=0.
  ///Currently the list of invalid patterns is hardcoded.
  ///This has to be read from configuration.
  virtual bool checkHitPatternValidity(unsigned int hits);

  std::unique_ptr<SorterBase<GoldenPatternType> > sorter;

  std::unique_ptr<IGhostBuster> ghostBuster;

};

#endif

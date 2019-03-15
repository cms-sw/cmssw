#ifndef OMTF_OMTFProcessorTTMerger_H
#define OMTF_OMTFProcessorTTMerger_H

#include <L1Trigger/L1TMuonBayes/interface/Omtf/GoldenPattern.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/GoldenPatternResult.h>
#include <memory>

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"

#include "SimDataFormats/Track/interface/SimTrack.h"

#include "L1Trigger/L1TMuonBayes/interface/Omtf/GoldenPattern.h"
#include <L1Trigger/L1TMuonBayes/interface/Omtf/IGhostBuster.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/IProcessorEmulator.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/OMTFConfiguration.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/OMTFProcessor.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/ProcessorBase.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/SorterBase.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/SorterBase.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/TTAlgoMuon.h>


class OMTFinput;

namespace edm{
class ParameterSet;
};

template <class GoldenPatternType>
class OMTFProcessorTTMerger: public OMTFProcessor<GoldenPatternType> {
 public:
  OMTFProcessorTTMerger(OMTFConfiguration* omtfConfig, const edm::ParameterSet& edmCfg, const edm::EventSetup & evSetup, const L1TMuonOverlapParams* omtfPatterns);

  OMTFProcessorTTMerger(OMTFConfiguration* omtfConfig, const edm::ParameterSet& edmCfg, const edm::EventSetup & evSetup, const typename ProcessorBase<GoldenPatternType>::GoldenPatternVec& gps);

  virtual ~OMTFProcessorTTMerger();

  ///Fill GP vec with patterns from CondFormats object
/*  virtual bool configure(const OMTFConfiguration* omtfParams, const L1TMuonOverlapParams* omtfPatterns) {
    return ProcessorBase<GoldenPatternType>::configure(omtfParams, omtfPatterns);
  }*/

  void laodTTTracks(const edm::Event &event, const edm::ParameterSet& edmCfg);

  TTTracks getTTTrackForProcessor(unsigned int iProcessor, l1t::tftype mtfType, const TTTracks& eventTTTRacks);

  virtual void loadAndFilterDigis(const edm::Event& iEvent, const edm::ParameterSet& edmCfg) {
    OMTFProcessor<GoldenPatternType>::loadAndFilterDigis(iEvent, edmCfg);
    laodTTTracks(iEvent, edmCfg);
  }

   ///Process input data from a single event
  ///Input data is represented by hits in logic layers expressed in local coordinates
  ///Vector index: number of the ref hit (from 0 to nTestRefHits i.e. 4)
  ///Map key: GoldenPattern key
  //const std::vector<OMTFProcessor::resultsMap> &
  virtual const void processInput(unsigned int iProcessor, l1t::tftype mtfType,
							      const OMTFinput& aInput, const TTTracks& ttTracks);
  
  virtual std::vector<l1t::RegionalMuonCand> run(unsigned int iProcessor, l1t::tftype mtfType, int bx, std::vector<std::unique_ptr<IOMTFEmulationObserver> >& observers);

  static int ghostBust1(std::shared_ptr<AlgoMuon> first, std::shared_ptr<AlgoMuon> second);
  static int ghostBust2(std::shared_ptr<AlgoMuon> first, std::shared_ptr<AlgoMuon> second);
  static int ghostBust3(std::shared_ptr<AlgoMuon> first, std::shared_ptr<AlgoMuon> second);

  void setGhostBustFunc(
      const std::function<
          int(std::shared_ptr<AlgoMuon> first,
              std::shared_ptr<AlgoMuon> second)>& ghostBustFunc) {
    this->ghostBustFunc = ghostBustFunc;
  }

  bool checkHitPatternValidity(unsigned int hits);
  std::vector<l1t::RegionalMuonCand> getFinalcandidates(unsigned int iProcessor, l1t::tftype mtfType, AlgoMuons& algoCand);

 protected:

 private:
  ///Check if the hit pattern of given OMTF candite is not on the list
  ///of invalid hit patterns. Invalid hit patterns provode very little
  ///to efficiency, but gives high contribution to rate.
  ///Candidate with invalid hit patterns is assigned quality=0.
  ///Currently the list of invalid patterns is hardcoded.
  ///This has to be read from configuration.
/*
  virtual bool checkHitPatternValidity(unsigned int hits);

  std::unique_ptr<SorterBase<GoldenPatternType> > sorter;

  std::unique_ptr<IGhostBuster> ghostBuster;
*/

  enum TTTracksSource {
    NONE,
    SIM_TRACKS,
    L1_TRACKER
  };

  TTTracksSource ttTracksSource = NONE;

  ///all ttTracks in a event
  TTTracks ttTracks;

  ///TTMuons before ghostbusting, altered by each the processInput() and run()
  ///thus should not be used outside the processor
  AlgoMuons ttMuons;

  ///TTMuons after ghostbusting, altered by each the processInput() and run()
  ///thus should not be used outside the processor
  AlgoMuons selectedTTMuons;

  void init(const edm::ParameterSet& edmCfg);

  /**should return:
   * 0 if first kills second
   * 1 if second kills first
   * 2 otherwise (none is killed)
   */
  std::function<int (std::shared_ptr<AlgoMuon> first, std::shared_ptr<AlgoMuon> second)> ghostBustFunc;


  int l1Tk_nPar = 4;

  ///for patterns generation this shpuld be false, for normal running - true
  bool refLayerMustBeValid = true;

  //for testing and debugging
  void modifyPatterns();
};

#endif

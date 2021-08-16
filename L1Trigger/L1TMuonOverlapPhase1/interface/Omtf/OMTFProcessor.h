#ifndef OMTF_OMTFProcessor_H
#define OMTF_OMTFProcessor_H

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/AlgoMuon.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GoldenPattern.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GoldenPatternResult.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/IGhostBuster.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/IProcessorEmulator.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFConfiguration.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFinputMaker.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/ProcessorBase.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/SorterBase.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/SorterBase.h"

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/PtAssignmentBase.h"

#include <memory>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GoldenPattern.h"

class OMTFinput;

namespace edm {
  class ParameterSet;
};

template <class GoldenPatternType>
class OMTFProcessor : public ProcessorBase<GoldenPatternType>, public IProcessorEmulator {
public:
  OMTFProcessor(OMTFConfiguration* omtfConfig,
                const edm::ParameterSet& edmCfg,
                edm::EventSetup const& evSetup,
                const L1TMuonOverlapParams* omtfPatterns);

  OMTFProcessor(OMTFConfiguration* omtfConfig,
                const edm::ParameterSet& edmCfg,
                edm::EventSetup const& evSetup,
                const typename ProcessorBase<GoldenPatternType>::GoldenPatternVec& gps);

  ~OMTFProcessor() override;

  ///Fill GP vec with patterns from CondFormats object
  /*  virtual bool configure(const OMTFConfiguration* omtfParams, const L1TMuonOverlapParams* omtfPatterns) {
    return ProcessorBase<GoldenPatternType>::configure(omtfParams, omtfPatterns);
  }*/

  ///Process input data from a single event
  ///Input data is represented by hits in logic layers expressed in local coordinates
  void processInput(unsigned int iProcessor, l1t::tftype mtfType, const OMTFinput& aInput) override;

  AlgoMuons sortResults(unsigned int iProcessor, l1t::tftype mtfType, int charge = 0) override;

  AlgoMuons ghostBust(AlgoMuons refHitCands, int charge = 0) override {
    return ghostBuster->select(refHitCands, charge);
  }

  //convert algo muon to outgoing Candidates
  std::vector<l1t::RegionalMuonCand> getFinalcandidates(unsigned int iProcessor,
                                                        l1t::tftype mtfType,
                                                        const AlgoMuons& algoCands) override;

  ///allows to use other sorter implementation than the default one
  virtual void setSorter(SorterBase<GoldenPatternType>* sorter) { this->sorter.reset(sorter); }

  ///allows to use other IGhostBuster implementation than the default one
  void setGhostBuster(IGhostBuster* ghostBuster) override { this->ghostBuster.reset(ghostBuster); }

  virtual void setPtAssignment(PtAssignmentBase* ptAssignment) { this->ptAssignment = ptAssignment; }

  std::vector<l1t::RegionalMuonCand> run(unsigned int iProcessor,
                                         l1t::tftype mtfType,
                                         int bx,
                                         OMTFinputMaker* inputMaker,
                                         std::vector<std::unique_ptr<IOMTFEmulationObserver> >& observers) override;

  void printInfo() const override;

private:
  virtual void init(const edm::ParameterSet& edmCfg, edm::EventSetup const& evSetup);

  ///Check if the hit pattern of given OMTF candite is not on the list
  ///of invalid hit patterns. Invalid hit patterns provode very little
  ///to efficiency, but gives high contribution to rate.
  ///Candidate with invalid hit patterns is assigned quality=0.
  ///Currently the list of invalid patterns is hardcoded.
  ///This has to be read from configuration.
  bool checkHitPatternValidity(unsigned int hits) override;

  std::unique_ptr<SorterBase<GoldenPatternType> > sorter;

  std::unique_ptr<IGhostBuster> ghostBuster;

  PtAssignmentBase* ptAssignment = nullptr;
  //should be destroyed where it is created, i.e. by OmtfEmulation or OMTFReconstruction
};

#endif

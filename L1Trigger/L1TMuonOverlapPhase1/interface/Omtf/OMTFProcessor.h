#ifndef L1T_OmtfP1_OMTFProcessor_H
#define L1T_OmtfP1_OMTFProcessor_H

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/AlgoMuon.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/FinalMuon.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GoldenPattern.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GoldenPatternResult.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/IGhostBuster.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/IProcessorEmulator.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFConfiguration.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFinputMaker.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/ProcessorBase.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/SorterBase.h"

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/PtAssignmentBase.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GoldenPattern.h"

#include <memory>
#include <map>

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
                GoldenPatternVec<GoldenPatternType>&& gps);

  ~OMTFProcessor() override;

  ///Fill GP vec with patterns from CondFormats object
  /*  virtual bool configure(const OMTFConfiguration* omtfParams, const L1TMuonOverlapParams* omtfPatterns) {
    return ProcessorBase<GoldenPatternType>::configure(omtfParams, omtfPatterns);
  }*/

  //targetStubQuality matters only for the barrel stubs,
  //targetStubR - radial distance from the z axis (beam), matters only for the endcap stubs
  //floating point version, it is used to generate the extrapolFactors for the fixed point version
  int extrapolateDtPhiBFloatPoint(const int& refLogicLayer,
                                  const int& refPhi,
                                  const int& refPhiB,
                                  const int& refHitSuperLayer,
                                  unsigned int targetLayer,
                                  const int& targetStubPhi,
                                  const int& targetStubQuality,
                                  const int& targetStubEta,
                                  const int& targetStubR,
                                  const OMTFConfiguration* omtfConfig);

  //fixed point, firmware like extrapolation
  int extrapolateDtPhiBFixedPoint(const int& refLogicLayer,
                                  const int& refPhi,
                                  const int& refPhiB,
                                  const int& refHitSuperLayer,
                                  unsigned int targetLayer,
                                  const int& targetStubPhi,
                                  const int& targetStubQuality,
                                  const int& targetStubEta,
                                  const int& targetStubR,
                                  const OMTFConfiguration* omtfConfig);

  int extrapolateDtPhiB(const MuonStubPtr& refStub,
                        const MuonStubPtr& targetStub,
                        unsigned int targetLayer,
                        const OMTFConfiguration* omtfConfig);

  ///Process input data from a single event
  ///Input data is represented by hits in logic layers expressed in local coordinates
  void processInput(unsigned int iProcessor,
                    l1t::tftype mtfType,
                    const OMTFinput& aInput,
                    std::vector<std::unique_ptr<IOMTFEmulationObserver> >& observers) override;

  AlgoMuons sortResults(unsigned int iProcessor, l1t::tftype mtfType, int charge = 0) override;

  AlgoMuons ghostBust(AlgoMuons refHitCands, int charge = 0) override {
    return ghostBuster->select(refHitCands, charge);
  }

  void assignQuality(AlgoMuons::value_type& algoMuon);

  FinalMuons getFinalMuons(unsigned int iProcessor, l1t::tftype mtfType, const AlgoMuons& gbCandidates) override;

  void convertToGmtScalesPhase1(unsigned int iProcessor, l1t::tftype mtfType, FinalMuonPtr& finalMuon);

  std::vector<l1t::RegionalMuonCand> getRegionalMuonCands(unsigned int iProcessor,
                                                          l1t::tftype mtfType,
                                                          FinalMuons& finalMuons) override;

  ///allows to use other sorter implementation than the default one
  virtual void setSorter(SorterBase<GoldenPatternType>* sorter) { this->sorter.reset(sorter); }

  ///allows to use other IGhostBuster implementation than the default one
  void setGhostBuster(IGhostBuster* ghostBuster) override { this->ghostBuster.reset(ghostBuster); }

  FinalMuons run(unsigned int iProcessor,
                 l1t::tftype mtfType,
                 int bx,
                 OMTFinputMaker* inputMaker,
                 std::vector<std::unique_ptr<IOMTFEmulationObserver> >& observers) override;

  void printInfo() const override;

  void saveExtrapolFactors();
  void loadExtrapolFactors(const std::string& filename);

private:
  virtual void init(const edm::ParameterSet& edmCfg, edm::EventSetup const& evSetup);

  ///Check if the hit pattern of given OMTF candite is not on the list
  ///of invalid hit patterns. Invalid hit patterns provode very little
  ///to efficiency, but gives high contribution to rate.
  ///Candidate with invalid hit patterns is assigned quality=0.
  ///Currently the list of invalid patterns is hardcoded.
  ///This has to be read from configuration.
  static bool checkHitPatternValidity(unsigned int hits);

  std::unique_ptr<SorterBase<GoldenPatternType> > sorter;

  std::unique_ptr<IGhostBuster> ghostBuster;

  bool useStubQualInExtr = false;
  bool useEndcapStubsRInExtr = false;

  //if true, the extrapolateDtPhiBFloatPoint, and the extrapolation factors are generated
  //if false, extrapolateDtPhiBFixedPoint is used
  bool useFloatingPointExtrapolation = false;

  int extrapolMultiplier = 128;

  std::vector<std::vector<std::map<int, double> > > extrapolFactors;  //[refLayer][targetLayer][etaCode]
  std::vector<std::vector<std::map<int, int> > > extrapolFactorsNorm;
};

#endif

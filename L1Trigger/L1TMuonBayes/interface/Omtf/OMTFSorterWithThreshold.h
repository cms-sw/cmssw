#ifndef OMTF_OMTFSorterWithThreshold_H
#define OMTF_OMTFSorterWithThreshold_H

#include <L1Trigger/L1TMuonBayes/interface/Omtf/GoldenPattern.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/SorterBase.h>
#include <vector>

template <class GoldenPatternType>
class OMTFSorterWithThreshold: public SorterBase<GoldenPatternType> {
public:
  enum Mode {
    bestGPByThresholdOnProbability2,
    bestGPByMaxGpProbability1
  };
  OMTFSorterWithThreshold(const OMTFConfiguration* myOmtfConfig, Mode mode) : myOmtfConfig(myOmtfConfig), mode(mode) {}

  virtual ~OMTFSorterWithThreshold() {}

  ///Sort results from a single reference hit.
  ///Select candidate with highest number of hit layers
  ///Then select a candidate with largest likelihood value and given charge
  ///as we allow two candidates with opposite charge from single 10deg region
  virtual AlgoMuons::value_type sortRefHitResults(unsigned int procIndx, unsigned int iRefHit, const std::vector< std::shared_ptr<GoldenPatternType> >& gPatterns,
				int charge=0);

private:
  const OMTFConfiguration* myOmtfConfig;

  Mode mode = bestGPByThresholdOnProbability2;

  std::vector< std::shared_ptr<GoldenPatternType> > gPatternsSortedByPt;
};

#endif

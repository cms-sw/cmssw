#ifndef OMTF_OMTFSorter_H
#define OMTF_OMTFSorter_H

#include <L1Trigger/L1TMuonBayes/interface/Omtf/SorterBase.h>
#include <vector>

template <class GoldenPatternType>
class OMTFSorter: public SorterBase<GoldenPatternType> {
public:
  virtual ~OMTFSorter() {}

  ///Sort results from a single reference hit.
  ///Select candidate with highest number of hit layers
  ///Then select a candidate with largest likelihood value and given charge
  ///as we allow two candidates with opposite charge from single 10deg region
  virtual AlgoMuons::value_type sortRefHitResults(unsigned int procIndx, unsigned int iRefHit, const std::vector< std::shared_ptr<GoldenPatternType> >& gPatterns,
				int charge=0);
};

#endif

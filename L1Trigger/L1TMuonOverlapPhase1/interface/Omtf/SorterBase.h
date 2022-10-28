/*
 * ISorter.h
 *
 *  Created on: Jun 28, 2017
 *      Author: kbunkow
 */

#ifndef L1T_OmtfP1_SORTERBASE_H_
#define L1T_OmtfP1_SORTERBASE_H_

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/AlgoMuon.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GoldenPatternResult.h"

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"

#include <vector>

template <class GoldenPatternType>
class SorterBase {
public:
  virtual ~SorterBase() {}

  //iProcessor - continuous processor index [0...11]
  virtual AlgoMuons sortResults(unsigned int procIndx,
                                const GoldenPatternVec<GoldenPatternType>& gPatterns,
                                int charge = 0) {
    AlgoMuons refHitCands;
    //  for(auto itRefHit: procResults) refHitCands.push_back(sortRefHitResults(itRefHit,charge));
    for (unsigned int iRefHit = 0; iRefHit < gPatterns.at(0)->getResults()[procIndx].size(); iRefHit++) {
      refHitCands.emplace_back(sortRefHitResults(procIndx, iRefHit, gPatterns, charge));
    }
    return refHitCands;
  }

  ///Sort results from a single reference hit.
  ///Select candidate with highest number of hit layers
  ///Then select a candidate with largest likelihood value and given charge
  ///as we allow two candidates with opposite charge from single 10deg region
  virtual AlgoMuons::value_type sortRefHitResults(unsigned int procIndx,
                                                  unsigned int iRefHit,
                                                  const GoldenPatternVec<GoldenPatternType>& gPatterns,
                                                  int charge = 0) = 0;
};

#endif /* L1T_OmtfP1_SORTERBASE_H_ */

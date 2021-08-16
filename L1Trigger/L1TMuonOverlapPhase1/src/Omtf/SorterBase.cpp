/*
 * ISorter.cpp
 *
 *  Created on: Jun 28, 2017
 *      Author: kbunkow
 */

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/SorterBase.h"

///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
/*
void SorterBase::sortResults(const std::vector<IGoldenPattern*>& gPatterns,
              std::vector<AlgoMuon> & refHitCands,
              int charge) {

//  for(auto itRefHit: procResults) refHitCands.push_back(sortRefHitResults(itRefHit,charge));
  for (unsigned int iRefHit = 0 ; iRefHit < gPatterns.at(0)->getResults()[procIndx].size(); iRefHit++) {
    AlgoMuon mu = sortRefHitResults(iRefHit, gPatterns, charge);
    mu.setRefHitNumber(iRefHit);
    refHitCands.push_back(mu);
  }
}
*/

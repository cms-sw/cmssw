/** \file
 *
 * $Date:  01/03/2006 16:59:11 CET $
 * $Revision: 1.0 $
 * \author : Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 */

/* This Class Header */
#include "RecoLocalMuon/DTSegment/src/DTSegmentCleaner.h"

/* Collaborating Class Header */

/* C++ Headers */
using namespace std;

/* ====================================================================== */

/* Constructor */ 
DTSegmentCleaner::DTSegmentCleaner(const edm::ParameterSet& pset) {
}

/* Destructor */ 
DTSegmentCleaner::~DTSegmentCleaner() {
}

/* Operations */ 
vector<DTSegmentCand*> DTSegmentCleaner::clean(vector<DTSegmentCand*> inputCands) const {
  if (inputCands.size()<2) return inputCands;

  vector<DTSegmentCand*> result = solveConflict(inputCands);

  // cout << "No conflict " << endl;
  // for (vector<DTSegmentCand*>::const_iterator seg=result.begin();
  //      seg!=result.end(); ++seg) 
  //   cout << *(*seg) << endl;

  result = ghostBuster(result);

  return result;
}

vector<DTSegmentCand*> DTSegmentCleaner::solveConflict(vector<DTSegmentCand*> inputCands) const {
  vector<DTSegmentCand*> result;

  vector<DTSegmentCand*> ghosts;
  for (vector<DTSegmentCand*>::iterator cand=inputCands.begin();
       cand!=inputCands.end(); ++cand) {
    for (vector<DTSegmentCand*>::iterator cand2=cand+1;
         cand2!=inputCands.end(); ++cand2) {
      DTSegmentCand::AssPointCont confHits=(*cand)->conflictingHitPairs(*(*cand2));
      if (confHits.size()) {
        for (DTSegmentCand::AssPointCont::const_iterator cHit=confHits.begin() ;
             cHit!=confHits.end(); ++cHit) {
          if ((**cand)<(**cand2)) 
            (*cand)->removeHit(*cHit);
          else  
            (*cand2)->removeHit(*cHit);
        }
      }
    }
  }

  vector<DTSegmentCand*>::iterator cand=inputCands.begin();
  while ( cand < inputCands.end() ) {
    if ((*cand)->good()) 
      result.push_back(*cand);
    else {
      vector<DTSegmentCand*>::iterator badCand=cand;
      delete *badCand;
    }
    ++cand;
  }

  return result;
}

vector<DTSegmentCand*> 
DTSegmentCleaner::ghostBuster(vector<DTSegmentCand*> inputCands) const {
  vector<DTSegmentCand*> result;
  vector<DTSegmentCand*> ghosts;
  for (vector<DTSegmentCand*>::iterator cand=inputCands.begin();
       cand!=inputCands.end(); ++cand) {
    for (vector<DTSegmentCand*>::iterator cand2=cand+1;
         cand2!=inputCands.end(); ++cand2) {
      int nSharedHits=(*cand)->nSharedHitPairs(*(*cand2));
      // cout << "Sharing " << (**cand) << " " << (**cand2) << " " << nSharedHits
      //   << " < " << ((**cand)<(**cand2)) << endl;
      if (nSharedHits >= nSharedHitsMax ) {
        if ((**cand)<(**cand2)) {
          // cout << (**cand) << " is ghost " << endl;
          ghosts.push_back(*cand);
        }
        else {
          // cout << (**cand2) << " is ghost " << endl;
          ghosts.push_back(*cand2);
        }
        continue;
      }
    }
  }

  for (vector<DTSegmentCand*>::const_iterator cand=inputCands.begin();
       cand!=inputCands.end(); ++cand) {
    bool isGhost=false;
    for (vector<DTSegmentCand*>::const_iterator ghost=ghosts.begin();
         ghost!=ghosts.end(); ++ghost) {
      if ((*cand)==(*ghost)) {
        isGhost=true;
        break;
      }
    }
    if (!isGhost) result.push_back(*cand);
    else delete *cand;
  }
  // cout << "No Ghosts ------" << endl;
  // for (vector<DTSegmentCand*>::iterator cand=result.begin();
  //      cand!=result.end(); ++cand) {
  //   cout << "cand " << *cand << " nH " <<(*cand)->nHits() << " chi2 " << (*cand)->chi2() << endl;
  // }
  // cout << "----------------" << endl;

  return result;
}

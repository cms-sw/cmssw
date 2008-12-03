/** \file
 *
 * $Date: 2007/02/13 13:31:40 $
 * $Revision: 1.8 $
 * \author : Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 */

/* This Class Header */
#include "RecoLocalMuon/DTSegment/src/DTSegmentCleaner.h"

/* Collaborating Class Header */
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

/* C++ Headers */
using namespace std;

/* ====================================================================== */

/* Constructor */ 
DTSegmentCleaner::DTSegmentCleaner(const edm::ParameterSet& pset) {
  nSharedHitsMax = pset.getParameter<int>("nSharedHitsMax");

  nUnSharedHitsMin = pset.getParameter<int>("nUnSharedHitsMin");

  segmCleanerMode = pset.getParameter<int>("segmCleanerMode");
 
  if((segmCleanerMode!=1)&&(segmCleanerMode!=2)&&(segmCleanerMode!=3))
    edm::LogError("Muon|RecoLocalMuon|DTSegmentCleaner")
      << "Wrong segmCleanerMode! It must be 1,2 or 3. The default is 1";
}

/* Destructor */ 
DTSegmentCleaner::~DTSegmentCleaner() {
}

/* Operations */ 
vector<DTSegmentCand*> DTSegmentCleaner::clean(vector<DTSegmentCand*> inputCands) const {
  if (inputCands.size()<2) return inputCands;
 
  vector<DTSegmentCand*> result = solveConflict(inputCands);

  result = ghostBuster(result);
  
  return result;
}

vector<DTSegmentCand*> DTSegmentCleaner::solveConflict(vector<DTSegmentCand*> inputCands) const {
  vector<DTSegmentCand*> result;

  vector<DTSegmentCand*> ghosts;

  for (vector<DTSegmentCand*>::iterator cand=inputCands.begin();
       cand!=inputCands.end(); ++cand) {
    for (vector<DTSegmentCand*>::iterator cand2 = cand+1 ; cand2!=inputCands.end() ; ++cand2) {

      DTSegmentCand::AssPointCont confHits=(*cand)->conflictingHitPairs(*(*cand2));

      if ( segmCleanerMode != 1
           && (confHits.size())==((*cand)->nHits()) && (confHits.size())==((*cand2)->nHits())
           && (fabs((*cand)->chi2()-(*cand2)->chi2())<0.1))
      {
        if(segmCleanerMode==2)
        {	
          LocalVector dir1 = (*cand)->direction();
          LocalVector dir2 = (*cand2)->direction();
          float phi1=(atan((dir1.x())/(dir1.z())));
          float phi2=(atan((dir2.x())/(dir2.z())));

          DTSegmentCand* badCand = (fabs(phi1) > fabs(phi2)) ? (*cand) : (*cand2);
          for (DTSegmentCand::AssPointCont::const_iterator cHit=confHits.begin() ;
               cHit!=confHits.end(); ++cHit) {
            badCand->removeHit(*cHit);
          }
        }
        else 
          LogDebug("Muon|RecoLocalMuon|DTSegmentCleaner")
            << "keep both segment candidates "<<*(*cand)<<" and "<<*(*cand2)<<endl;
      }
      else if (confHits.size()) {
        DTSegmentCand* badCand = (**cand) < (**cand2) ? (*cand) : (*cand2);
        for (DTSegmentCand::AssPointCont::const_iterator cHit=confHits.begin() ;
             cHit!=confHits.end(); ++cHit) {
          badCand->removeHit(*cHit);
        }
      }
      // HERE I should delete the worst segment if is not good enough
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
  vector<DTSegmentCand*> ghosts;
  for (vector<DTSegmentCand*>::iterator cand=inputCands.begin();
       cand!=inputCands.end(); ++cand) {
    for (vector<DTSegmentCand*>::iterator cand2=cand+1;
         cand2!=inputCands.end(); ++cand2) {
      unsigned int nSharedHits=(*cand)->nSharedHitPairs(*(*cand2));
      // cout << "Sharing " << (**cand) << " " << (**cand2) << " " << nSharedHits
      //   << " (first or second) " << ((**cand)<(**cand2)) << endl;
      if ((nSharedHits==((*cand)->nHits())) && (nSharedHits==((*cand2)->nHits()))
          &&(fabs((*cand)->chi2()-(*cand2)->chi2())<0.1)
          &&(segmCleanerMode==3))
      {
        continue;
      }

      // remove the worst segment if too many shared hits or too few unshared
      if ((int)nSharedHits >= nSharedHitsMax ||
          (int)((*cand)->nHits()-nSharedHits)<=nUnSharedHitsMin ||
          (int)((*cand2)->nHits()-nSharedHits)<=nUnSharedHitsMin) {

        if ((**cand)<(**cand2)) {
          ghosts.push_back(*cand);
        }
        else {
          ghosts.push_back(*cand2);
        }
        continue;
      }

    }
  }

  vector<DTSegmentCand*> result;
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

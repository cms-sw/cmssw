/** \file
 *
 * $Date: 2006/08/15 10:29:22 $
 * $Revision: 1.5 $
 * \author : Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 */

/* This Class Header */
#include "RecoLocalMuon/DTSegment/src/DTSegmentCleaner.h"

/* Collaborating Class Header */
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/* C++ Headers */
using namespace std;

/* ====================================================================== */

/* Constructor */ 
DTSegmentCleaner::DTSegmentCleaner(const edm::ParameterSet& pset) {
  nSharedHitsMax = pset.getParameter<int>("nSharedHitsMax");
  segmCleanerMode = pset.getParameter<int>("segmCleanerMode");
  if((segmCleanerMode!=1)&&(segmCleanerMode!=2)&&(segmCleanerMode!=3))
    {
      cout<<"Warning: wrong segmCleanerMode "<<segmCleanerMode
	  <<"    ->     default (1) has been chosen"<<endl;
      segmCleanerMode=1;
    }
}

/* Destructor */ 
DTSegmentCleaner::~DTSegmentCleaner() {
}

/* Operations */ 
vector<DTSegmentCand*> DTSegmentCleaner::clean(vector<DTSegmentCand*> inputCands) const {
  if (inputCands.size()<2) return inputCands;

 
  vector<DTSegmentCand*> result = solveConflict(inputCands);

  /*cout << "No conflict --------------" << endl;
  for (vector<DTSegmentCand*>::const_iterator seg=result.begin();
       seg!=result.end(); ++seg) {
    cout << *(*seg) << endl;
      AssPointCont myAssPointCont = (*seg)->hits();
      for (set<AssPoint, DTSegmentCand::AssPointLessZ>::const_iterator hits=myAssPointCont.begin();
	   hits!=myAssPointCont.end(); ++hits) 
	{	
	  cout<<(*hits).second<<endl;
	  cout<<(*hits).first->id()<<endl;
	}
	}*/
  
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
      if ((confHits.size())==((*cand)->nHits()) && (confHits.size())==((*cand2)->nHits())
	  &&(fabs((*cand)->chi2()-(*cand2)->chi2())<0.1)
	  &&(segmCleanerMode!=1))
	{
	  if(segmCleanerMode==2)
	    {	
	      LocalVector dir1 = (*cand)->direction();
	      LocalVector dir2 = (*cand2)->direction();
	      float phi1=(atan((dir1.x())/(dir1.z())));
	      float phi2=(atan((dir2.x())/(dir2.z())));
	      
	      if(fabs(phi1)>fabs(phi2))
		for (DTSegmentCand::AssPointCont::const_iterator cHit=confHits.begin() ;
		     cHit!=confHits.end(); ++cHit) {
		  (*cand)->removeHit(*cHit);
		}
	      else
		for (DTSegmentCand::AssPointCont::const_iterator cHit=confHits.begin() ;
		     cHit!=confHits.end(); ++cHit) {
		  (*cand2)->removeHit(*cHit);
		}
	    }
	  else 
	    cout<<"keep both segment candidates "<<*(*cand)<<" and "<<*(*cand2)<<endl;
	}
      else if (confHits.size()) {
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
	 unsigned int nSharedHits=(*cand)->nSharedHitPairs(*(*cand2));
       // cout << "Sharing " << (**cand) << " " << (**cand2) << " " << nSharedHits
       //   << " < " << ((**cand)<(**cand2)) << endl;
       	 if ((nSharedHits==((*cand)->nHits())) && (nSharedHits==((*cand2)->nHits()))
	  &&(fabs((*cand)->chi2()-(*cand2)->chi2())<0.1)
	  &&(segmCleanerMode==3))
	 {
	   continue;
	 }
	 if ((int)nSharedHits >= nSharedHitsMax ) {
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

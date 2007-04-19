#include "PhysicsTools/JetMCAlgos/interface/CandOneToOneDeltaRMatcher.h"
#include "PhysicsTools/JetMCUtils/interface/combination.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"


#include <Math/VectorUtil.h>
#include <TMath.h>

using namespace edm;
using namespace std;
using namespace reco;
using namespace ROOT::Math::VectorUtil;
using namespace stdcomb;

CandOneToOneDeltaRMatcher::CandOneToOneDeltaRMatcher( const ParameterSet & cfg ) :
  source_( cfg.getParameter<InputTag>( "src" ) ),
  matched_( cfg.getParameter<InputTag>( "matched" ) ),
  printdebug_( cfg.getUntrackedParameter<bool>("printDebug", false) ) {
  produces<CandMatchMap>();
}

CandOneToOneDeltaRMatcher::~CandOneToOneDeltaRMatcher() {
}
		
void CandOneToOneDeltaRMatcher::produce( Event& evt, const EventSetup& es ) {
  
  Handle<CandidateCollection> source;  
  Handle<CandidateCollection> matched;  
  evt.getByLabel( source_, source ) ;
  evt.getByLabel( matched_, matched ) ;
 
  if (printdebug_) {
    for( CandidateCollection::const_iterator c = source->begin(); c != source->end(); ++c ) {
      cout << "[CandOneToOneDeltaRMatcher] Et source  " << c->et() << endl;
    }    
    for( CandidateCollection::const_iterator c = matched->begin(); c != matched->end(); ++c ) {
      cout << "[CandOneToOneDeltaRMatcher] Et matched " << c->et() << endl;
    } 
  }
 
  if( matched->size() > 9 || matched->size() <= 0 ) return;
  if( source->size()  > 9 || source->size()  <= 0 ) return;

  float totalDeltaR=0;
  float BestTotalDeltaR=100;

  const int nMin = min( source->size() , matched->size() );
  const int nMax = max( source->size() , matched->size() );

  if( source->size() <= matched->size() ) {
    for(CandidateCollection::const_iterator iSr  = source->begin();
	iSr != source->end();
	iSr++) {
      vector <float> tempAllDist;
      for(CandidateCollection::const_iterator iMt  = matched->begin();
	  iMt != matched->end();
	  iMt++) { 
	tempAllDist.push_back(DeltaR( iSr->p4() , iMt->p4() ) );
      }
      AllDist.push_back(tempAllDist);
      tempAllDist.clear();
    } 
  } else {
    for(CandidateCollection::const_iterator iMt  = matched->begin();
	iMt != matched->end();
	iMt++) {
      vector <float> tempAllDist;
      for(CandidateCollection::const_iterator iSr  = source->begin();
	  iSr != source->end();
	  iSr++) { 
	tempAllDist.push_back(DeltaR( iSr->p4() , iMt->p4() ) );
      }
      AllDist.push_back(tempAllDist);
      tempAllDist.clear();
    } 
  }
  
  if (printdebug_) {
    for(int m0=0; m0<nMin; m0++) {
      for(int m1=0; m1<nMax; m1++) {
	printf("%5.3f ",AllDist[m0][m1]);
      }
      cout << endl;
    }
  }
  
  vector<int> ca;
  vector<int> cb;
  vector<int> bestCB;

  for(int i1=0; i1<nMax; i1++) ca.push_back(i1);
  for(int i1=0; i1<nMin; i1++) cb.push_back(i1);

  do
  {
    //do your processing on the new combination here
    for(int cnt=0;cnt<TMath::Factorial(nMin); cnt++)
    {
       totalDeltaR = lenght(cb);
       if ( totalDeltaR < BestTotalDeltaR ) {
         BestTotalDeltaR = totalDeltaR;
         bestCB=cb;
       }
       next_permutation( cb.begin() , cb.end() );
    }
  }
  while(next_combination( ca.begin() , ca.end() , cb.begin() , cb.end() ));
  
  if (printdebug_) {
    cout << "[CandOneToOneDeltaRMatcher] Best DeltaR=" << BestTotalDeltaR << " ";
    for(vector<int>::iterator it=bestCB.begin(); it!=bestCB.end(); it++ ) cout << *it;
    cout << endl;
  }

  auto_ptr<CandMatchMap> matchMap( new CandMatchMap( CandMatchMap::ref_type( CandidateRefProd( source  ),
                                                                             CandidateRefProd( matched )  ) ) );

  for( int c = 0; c != nMin; c ++ ) {
    if( source->size() <= matched->size() ) { 
      matchMap->insert( CandidateRef( source, c   ), CandidateRef( matched, bestCB[c] ) ); 
    } else {
      matchMap->insert( CandidateRef( source, bestCB[c] ), CandidateRef( matched, c   ) );
    }
  }

  evt.put( matchMap );

  AllDist.clear();
}


double CandOneToOneDeltaRMatcher::lenght(vector<int> best) {
  double myLenght=0;
  int row=0;
  for(vector<int>::iterator it=best.begin(); it!=best.end(); it++ ) { 		
    myLenght+=AllDist[row][*it];
    row++;
  }
  return myLenght;
}

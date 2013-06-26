/* \class CandOneToOneDeltaRMatcher
 *
 * Producer for simple match map
 * to match two collections of candidate
 * with one-to-One matching 
 * minimizing Sum(DeltaR)
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include<vector>
#include<iostream>

class CandOneToOneDeltaRMatcher : public edm::EDProducer {
 public:
  CandOneToOneDeltaRMatcher( const edm::ParameterSet & );
  ~CandOneToOneDeltaRMatcher();
 private:
  void produce( edm::Event&, const edm::EventSetup& );
  double lenght( const std::vector<int>& );
  std::vector<int> AlgoBruteForce(int, int);
  std::vector<int> AlgoSwitchMethod(int, int);
  
  edm::InputTag source_;
  edm::InputTag matched_;
  std::vector < std::vector<float> > AllDist;
  std::string algoMethod_;

};

#include "PhysicsTools/JetMCUtils/interface/combination.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
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
  algoMethod_( cfg.getParameter<string>( "algoMethod" ) ) {
  produces<CandViewMatchMap>("src2mtc");
  produces<CandViewMatchMap>("mtc2src");
}

CandOneToOneDeltaRMatcher::~CandOneToOneDeltaRMatcher() {
}
		
void CandOneToOneDeltaRMatcher::produce( Event& evt, const EventSetup& es ) {
  
  Handle<CandidateView> source;  
  Handle<CandidateView> matched;  
  evt.getByLabel( source_, source ) ;
  evt.getByLabel( matched_, matched ) ;
 
  edm::LogVerbatim("CandOneToOneDeltaRMatcher") << "======== Source Collection =======";
  for( CandidateView::const_iterator c = source->begin(); c != source->end(); ++c ) {
    edm::LogVerbatim("CandOneToOneDeltaRMatcher") << " pt source  " << c->pt() << " " << c->eta() << " " << c->phi()  << endl;
  }    
  edm::LogVerbatim("CandOneToOneDeltaRMatcher") << "======== Matched Collection =======";
  for( CandidateView::const_iterator c = matched->begin(); c != matched->end(); ++c ) {
    edm::LogVerbatim("CandOneToOneDeltaRMatcher") << " pt source  " << c->pt() << " " << c->eta() << " " << c->phi()  << endl;
  } 

  const int nSrc = source->size();
  const int nMtc = matched->size();

  const int nMin = min( source->size() , matched->size() );
  const int nMax = max( source->size() , matched->size() );
  if( nMin < 1 ) return;

  if( nSrc <= nMtc ) {
    for(CandidateView::const_iterator iSr  = source->begin();
	iSr != source->end();
	iSr++) {
      vector <float> tempAllDist;
      for(CandidateView::const_iterator iMt  = matched->begin();
	  iMt != matched->end();
	  iMt++) { 
	tempAllDist.push_back(DeltaR( iSr->p4() , iMt->p4() ) );
      }
      AllDist.push_back(tempAllDist);
      tempAllDist.clear();
    } 
  } else {
    for(CandidateView::const_iterator iMt  = matched->begin();
	iMt != matched->end();
	iMt++) {
      vector <float> tempAllDist;
      for(CandidateView::const_iterator iSr  = source->begin();
	  iSr != source->end();
	  iSr++) { 
	tempAllDist.push_back(DeltaR( iSr->p4() , iMt->p4() ) );
      }
      AllDist.push_back(tempAllDist);
      tempAllDist.clear();
    } 
  }
  
  /*
  edm::LogVerbatim("CandOneToOneDeltaRMatcher") << "======== The DeltaR Matrix =======";
  for(int m0=0; m0<nMin; m0++) {
    //    for(int m1=0; m1<nMax; m1++) {
      edm::LogVerbatim("CandOneToOneDeltaRMatcher") << setprecision(2) << fixed << (m1 AllDist[m0][m1] ;
    //}
    edm::LogVerbatim("CandOneToOneDeltaRMatcher") << "\n"; 
  }
  */
  
  // Loop size if Brute Force
  int nLoopToDo = (int) ( TMath::Factorial(nMax) / TMath::Factorial(nMax - nMin) );
  edm::LogVerbatim("CandOneToOneDeltaRMatcher") << "nLoop:" << nLoopToDo << endl;
  edm::LogVerbatim("CandOneToOneDeltaRMatcher") << "Choosen Algo is:" << algoMethod_ ;
  vector<int> bestCB;

  // Algo is Brute Force
  if( algoMethod_ == "BruteForce") {

    bestCB = AlgoBruteForce(nMin,nMax);

  // Algo is Switch Method
  } else if( algoMethod_ == "SwitchMode" ) {

    bestCB = AlgoSwitchMethod(nMin,nMax);

  // Algo is Brute Force if nLoop < 10000
  } else if( algoMethod_ == "MixMode" ) {

    if( nLoopToDo < 10000 ) {
      bestCB = AlgoBruteForce(nMin,nMax);
    } else { 
      bestCB = AlgoSwitchMethod(nMin,nMax);
    } 

  } else {
    throw cms::Exception("OneToOne Constructor") << "wrong matching method in ParameterSet";
  }

  for(int i1=0; i1<nMin; i1++) edm::LogVerbatim("CandOneToOneDeltaRMatcher") << "min: " << i1 << " " << bestCB[i1] << " " << AllDist[i1][bestCB[i1]];

/*
  auto_ptr<CandViewMatchMap> matchMapSrMt( new CandViewMatchMap( CandViewMatchMap::ref_type( CandidateRefProd( source  ),
                                                                                             CandidateRefProd( matched )  ) ) );
  auto_ptr<CandViewMatchMap> matchMapMtSr( new CandViewMatchMap( CandViewMatchMap::ref_type( CandidateRefProd( matched ),
                                                                                             CandidateRefProd( source  )  ) ) );
*/

  auto_ptr<CandViewMatchMap> matchMapSrMt( new CandViewMatchMap() );
  auto_ptr<CandViewMatchMap> matchMapMtSr( new CandViewMatchMap() );

  for( int c = 0; c != nMin; c ++ ) {
    if( source->size() <= matched->size() ) {
      matchMapSrMt->insert( source ->refAt(c         ), matched->refAt(bestCB[c] ) );
      matchMapMtSr->insert( matched->refAt(bestCB[c] ), source ->refAt(c         ) );
    } else {
      matchMapSrMt->insert( source ->refAt(bestCB[c] ), matched->refAt(c         ) );
      matchMapMtSr->insert( matched->refAt(c         ), source ->refAt(bestCB[c] ) );
    }
  }

/*
  for( int c = 0; c != nMin; c ++ ) {
    if( source->size() <= matched->size() ) { 
      matchMapSrMt->insert( CandidateRef( source,  c         ), CandidateRef( matched, bestCB[c] ) ); 
      matchMapMtSr->insert( CandidateRef( matched, bestCB[c] ), CandidateRef( source, c          ) );
    } else {
      matchMapSrMt->insert( CandidateRef( source,  bestCB[c] ), CandidateRef( matched, c         ) );
      matchMapMtSr->insert( CandidateRef( matched, c         ), CandidateRef( source,  bestCB[c] ) );
    }
  }
*/
  evt.put( matchMapSrMt, "src2mtc" );
  evt.put( matchMapMtSr, "mtc2src" );

  AllDist.clear();
}


double CandOneToOneDeltaRMatcher::lenght(const vector<int>& best) {
  double myLenght=0;
  int row=0;
  for(vector<int>::const_iterator it=best.begin(); it!=best.end(); it++ ) { 		
    myLenght+=AllDist[row][*it];
    row++;
  }
  return myLenght;
}

// this is the Brute Force Algorithm
// All the possible combination are checked
// The best one is always found
// Be carefull when you have high values for nMin and nMax --> the combinatorial could explode!
// Sum(DeltaR) is minimized -->
// 0.1 - 0.2 - 1.0 - 1.5 is lower than
// 0.1 - 0.2 - 0.3 - 3.0 
// Which one do you prefer? --> BruteForce select always the first

vector<int> CandOneToOneDeltaRMatcher::AlgoBruteForce( int nMin, int nMax ) {

  vector<int> ca;
  vector<int> cb;
  vector<int> bestCB;
  float totalDeltaR=0;
  float BestTotalDeltaR=1000;

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
  
  return bestCB;
}

// This method (Developed originally by Daniele Benedetti) check for the best combination
// choosing the minimum DeltaR for each line in AllDist matrix
// If no repeated row is found: ie (line,col)=(1,3) and (2,3) --> same as BruteForce
// If repetition --> set the higher DeltaR between  the 2 repetition to 1000 and re-check best combination
// Iterate until no repetition  
// No guaranted minimum for Sum(DeltaR)
// If you have:
// 0.1 - 0.2 - 1.0 - 1.5 is lower than
// 0.1 - 0.2 - 0.3 - 3.0 
// SwitchMethod normally select the second solution

vector<int> CandOneToOneDeltaRMatcher::AlgoSwitchMethod( int nMin, int nMax ) {

  vector<int> bestCB;
  for(int i1=0; i1<nMin; i1++) {
    int minInd=0;
    for(int i2=1; i2<nMax; i2++) if( AllDist[i1][i2] < AllDist[i1][minInd] ) minInd = i2; 
    bestCB.push_back(minInd);
  }

  bool inside = true;
  while( inside ) {
    inside = false;
    for(int i1=0;i1<nMin;i1++){
      for(int i2=i1+1;i2<nMin;i2++){
	if ( bestCB[i1] == bestCB[i2] ) {
	  inside = true;
	  if ( AllDist[i1][(bestCB[i1])] <= AllDist[i2][(bestCB[i2])]) {
	    AllDist[i2][(bestCB[i2])]= 1000;
	    int minInd=0;
	    for(int i3=1; i3<nMax; i3++) if( AllDist[i2][i3] < AllDist[i2][minInd] ) minInd = i3; 
	    bestCB[i2]= minInd;
	  }  else {
	    AllDist[i1][(bestCB[i1])]= 1000;
	    int minInd=0;
	    for(int i3=1; i3<nMax; i3++) if( AllDist[i1][i3] < AllDist[i1][minInd] ) minInd = i3; 
	    bestCB[i1]= minInd;
	  }
	} // End if
      } 
    }
  } // End while

  return bestCB;

}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( CandOneToOneDeltaRMatcher );

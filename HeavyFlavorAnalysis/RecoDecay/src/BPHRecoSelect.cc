/*
 *  See header file for a description of this class.
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoSelect.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"

//---------------
// C++ Headers --
//---------------
#include <map>

using namespace std;

//-------------------
// Initializations --
//-------------------


//----------------
// Constructors --
//----------------
BPHRecoSelect::BPHRecoSelect() {
}

//--------------
// Destructor --
//--------------
BPHRecoSelect::~BPHRecoSelect() {
}

//--------------
// Operations --
//--------------
bool BPHRecoSelect::accept( const reco::Candidate& cand ) const {
  return true;
}


bool BPHRecoSelect::accept( const reco::Candidate& cand,
                            const BPHRecoBuilder* build ) const {
  return accept( cand );
}


const reco::Candidate* BPHRecoSelect::get( const string& name,
                                           const BPHRecoBuilder* build ) const {
  if ( build == 0 ) return 0;
  map<string,const reco::Candidate*>& cMap = build->daugMap;
  map<string,const reco::Candidate*>::iterator iter = cMap.find( name );
  return ( iter != cMap.end() ? iter->second : 0 );
}


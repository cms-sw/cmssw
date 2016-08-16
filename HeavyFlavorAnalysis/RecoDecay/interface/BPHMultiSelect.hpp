/*
 *  See header file for a description of this class.
 *
 *  $Date: 2015-07-06 18:40:19 $
 *  $Revision: 1.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
//#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHMultiSelect.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoSelect.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHMomentumSelect.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHVertexSelect.h"

//---------------
// C++ Headers --
//---------------


//-------------------
// Initializations --
//-------------------


//----------------
// Constructors --
//----------------
template<class T>
BPHMultiSelect<T>::BPHMultiSelect( BPHSelectOperation::mode op ) {
  switch ( op ) {
  case BPHSelectOperation:: or_mode:
    breakValue =  true;
    finalValue = false;
    break;
  case BPHSelectOperation::and_mode:
    breakValue = false;
    finalValue =  true;
    break;
  }
}

//--------------
// Destructor --
//--------------
template<class T>
BPHMultiSelect<T>::~BPHMultiSelect() {
}

//--------------
// Operations --
//--------------
template<class T>
void BPHMultiSelect<T>::include( T& s, bool m ) {
  SelectElement e;
  e.selector = &s;
  e.mode     = m;
  selectList.push_back( e );
  return;
}


template<class T>
bool BPHMultiSelect<T>::accept( const reco::Candidate& cand ) const {
  return false;
}


template<class T>
bool BPHMultiSelect<T>::accept( const reco::Candidate& cand,
                                const BPHRecoBuilder* build ) const {
  return false;
}


template<class T>
bool BPHMultiSelect<T>::accept( const BPHDecayMomentum& cand ) const {
  return false;
}


template<class T>
bool BPHMultiSelect<T>::accept( const BPHDecayVertex& cand ) const {
  return false;
}


template<>
bool BPHMultiSelect<BPHRecoSelect>::accept(
                                      const reco::Candidate& cand,
                                      const BPHRecoBuilder* build ) const;
template<>
bool BPHMultiSelect<BPHRecoSelect>::accept(
                                      const reco::Candidate& cand ) const;
template<>
bool BPHMultiSelect<BPHMomentumSelect>::accept(
                                          const BPHDecayMomentum& cand ) const;
template<>
bool BPHMultiSelect<BPHVertexSelect>::accept(
                                        const BPHDecayVertex& cand ) const;


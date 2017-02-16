#ifndef HeavyFlavorAnalysis_RecoDecay_BPHMultiSelect_h
#define HeavyFlavorAnalysis_RecoDecay_BPHMultiSelect_h
/** \class BPHMultiSelect
 *
 *  Description: 
 *     Class to combine multiple selection (OR mode)
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------


//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoSelect.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHMomentumSelect.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHVertexSelect.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHFitSelect.h"
class BPHRecoBuilder;
class BPHDecayMomentum;
class BPHDecayVertex;

namespace reco {
  class Candidate;
}

//---------------
// C++ Headers --
//---------------
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHSelectOperation {
 public:
  enum mode { or_mode, and_mode };
};

template<class T>
class BPHMultiSelect: public T {

 public:

  /** Constructor
   */
  BPHMultiSelect( BPHSelectOperation::mode op ) {
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

  /** Destructor
   */
  virtual ~BPHMultiSelect() {}

  /** Operations
   */
  /// include selection
  void include( T& s, bool m = true ) {
    SelectElement e;
    e.selector = &s;
    e.mode     = m;
    selectList.push_back( e );
    return;
  }

  /// accept function
  virtual bool accept( const reco::Candidate & cand,
                       const BPHRecoBuilder*  build ) const { return false; }
  virtual bool accept( const reco::Candidate & cand ) const { return false; }
  virtual bool accept( const BPHDecayMomentum& cand ) const { return false; }
  virtual bool accept( const BPHDecayVertex  & cand ) const { return false; }
  virtual bool accept( const BPHKinematicFit & cand ) const { return false; }

 private:

  // private copy and assigment constructors
  BPHMultiSelect           ( const BPHMultiSelect<T>& x );
  BPHMultiSelect& operator=( const BPHMultiSelect<T>& x );

  struct SelectElement {
    T* selector;
    bool mode;
  };

  bool breakValue;
  bool finalValue;
  std::vector<SelectElement> selectList;

  template<class Obj> bool select( const Obj& cand ) const {
    int i;
    int n = selectList.size();
    for ( i = 0; i < n; ++i ) {
      const SelectElement& e = selectList[i];
      if ( ( e.selector->accept( cand ) == e.mode ) == breakValue )
                                                return breakValue;
    }
    return finalValue;
  }
  template<class Obj> bool select( const Obj& cand,
                                   const BPHRecoBuilder* build ) const {
    int i;
    int n = selectList.size();
    for ( i = 0; i < n; ++i ) {
      const SelectElement& e = selectList[i];
      if ( ( e.selector->accept( cand, build ) == e.mode ) == breakValue )
                                                       return breakValue;
    }
    return finalValue;
  }

};

template<>
bool BPHMultiSelect<BPHRecoSelect    >::accept(
                                        const reco::Candidate & cand,
                                        const BPHRecoBuilder* build ) const;
template<>
bool BPHMultiSelect<BPHRecoSelect    >::accept(
                                        const reco::Candidate & cand ) const;
template<>
bool BPHMultiSelect<BPHMomentumSelect>::accept(
                                        const BPHDecayMomentum& cand ) const;
template<>
bool BPHMultiSelect<BPHVertexSelect  >::accept(
                                        const BPHDecayVertex  & cand ) const;
template<>
bool BPHMultiSelect<BPHFitSelect     >::accept(
                                        const BPHKinematicFit & cand ) const;

#endif


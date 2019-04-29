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


//BPHMultiSelectBase has the implementation needed for accept
// but does not itself override accept
template<class T>
class BPHMultiSelectBase : public T {

 public:
  /** Constructor
   */
  BPHMultiSelectBase( BPHSelectOperation::mode op ) {
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
  ~BPHMultiSelectBase() override {}

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

 protected:
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

 private:

  // private copy and assigment constructors
  BPHMultiSelectBase<T>           ( const BPHMultiSelectBase<T>& x ) = delete;
  BPHMultiSelectBase<T>& operator=( const BPHMultiSelectBase<T>& x ) = delete;

  struct SelectElement {
    T* selector;
    bool mode;
  };

  bool breakValue;
  bool finalValue;
  std::vector<SelectElement> selectList;

};

template<class T>
class BPHMultiSelect: public BPHMultiSelectBase<T> {

 public:
  using Base = BPHMultiSelectBase<T>;

  /** Constructor
   */
  BPHMultiSelect( BPHSelectOperation::mode op ):
  Base(op) {}

  /** Destructor
   */
  ~BPHMultiSelect() override {}

  /// accept function
  bool accept( const typename T::AcceptArg & cand ) const override;

 private:

  // private copy and assigment constructors
  BPHMultiSelect           ( const BPHMultiSelect<T>& x ) = delete;
  BPHMultiSelect& operator=( const BPHMultiSelect<T>& x ) = delete;
};

template<>
class BPHMultiSelect<BPHRecoSelect>: public BPHMultiSelectBase<BPHRecoSelect> {

 public:
  using Base = BPHMultiSelectBase<BPHRecoSelect>;

  /** Constructor
   */
  BPHMultiSelect( BPHSelectOperation::mode op ):
  Base(op) {}

  /** Destructor
   */
  ~BPHMultiSelect() override {}

  /// accept function
  bool accept( const typename BPHRecoSelect::AcceptArg & cand ) const override;
  bool accept( const reco::Candidate & cand, //NOLINT
	       const BPHRecoBuilder*  build ) const override; //NOLINT

 private:

  // private copy and assigment constructors
  BPHMultiSelect           ( const BPHMultiSelect<BPHRecoSelect>& x ) = delete;
  BPHMultiSelect& operator=( const BPHMultiSelect<BPHRecoSelect>& x ) = delete;
};

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


#ifndef BPHMultiSelect_H
#define BPHMultiSelect_H
/** \class BPHMultiSelect
 *
 *  Description: 
 *     Class to combine multiple selection (OR mode)
 *
 *
 *  $Date: 2015-07-06 18:40:19 $
 *  $Revision: 1.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------


//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
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
  BPHMultiSelect( BPHSelectOperation::mode op );

  /** Destructor
   */
  virtual ~BPHMultiSelect();

  /** Operations
   */
  /// include selection
  void include( T& s, bool m = true );

  /// accept function
  virtual bool accept( const reco::Candidate & cand,
                       const BPHRecoBuilder*  build ) const;
  virtual bool accept( const reco::Candidate & cand ) const;
  virtual bool accept( const BPHDecayMomentum& cand ) const;
  virtual bool accept( const BPHDecayVertex  & cand ) const;

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

#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHMultiSelect.hpp"

#endif // BPHMultiSelect_H


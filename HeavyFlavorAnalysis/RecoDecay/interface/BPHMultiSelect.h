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

template <class T>
class BPHMultiSelectBase : public T {
public:
  /** Constructor
   */
  BPHMultiSelectBase(BPHSelectOperation::mode op) {
    switch (op) {
      case BPHSelectOperation::or_mode:
        breakValue = true;
        finalValue = false;
        break;
      case BPHSelectOperation::and_mode:
        breakValue = false;
        finalValue = true;
        break;
    }
  }

  // deleted copy constructor and assignment operator
  BPHMultiSelectBase(const BPHMultiSelectBase<T>& x) = delete;
  BPHMultiSelectBase& operator=(const BPHMultiSelectBase<T>& x) = delete;

  /** Destructor
   */
  ~BPHMultiSelectBase() override = default;

  /** Operations
   */
  /// include selection
  void include(T& s, bool m = true) {
    SelectElement e;
    e.selector = &s;
    e.mode = m;
    selectList.push_back(e);
    return;
  }

  /// component count
  unsigned int count() { return selectList.size(); }

protected:
  using Obj = typename T::AcceptArg;
  bool select(const Obj& cand) const {
    int i;
    int n = selectList.size();
    for (i = 0; i < n; ++i) {
      const SelectElement& e = selectList[i];
      if ((e.selector->accept(cand) == e.mode) == breakValue)
        return breakValue;
    }
    return finalValue;
  }
  bool select(const Obj& cand, const BPHRecoBuilder* build) const {
    int i;
    int n = selectList.size();
    for (i = 0; i < n; ++i) {
      const SelectElement& e = selectList[i];
      if ((e.selector->accept(cand, build) == e.mode) == breakValue)
        return breakValue;
    }
    return finalValue;
  }

private:
  struct SelectElement {
    T* selector;
    bool mode;
  };

  bool breakValue;
  bool finalValue;
  std::vector<SelectElement> selectList;
};

template <class T>
class BPHSlimSelect : public BPHMultiSelectBase<T> {
public:
  using Base = BPHMultiSelectBase<T>;

  /** Constructor
   */
  BPHSlimSelect(BPHSelectOperation::mode op) : Base(op) {}

  // deleted copy constructor and assignment operator
  BPHSlimSelect(const BPHSlimSelect<T>& x) = delete;
  BPHSlimSelect& operator=(const BPHSlimSelect<T>& x) = delete;

  /** Destructor
   */
  ~BPHSlimSelect() override = default;

  /** Operations
   */
  /// accept function
  bool accept(const typename T::AcceptArg& cand) const override { return Base::select(cand); }
};

template <class T>
class BPHFullSelect : public BPHSlimSelect<T> {
public:
  using Base = BPHSlimSelect<T>;

  /** Constructor
   */
  BPHFullSelect(BPHSelectOperation::mode op) : Base(op) {}

  // deleted copy constructor and assignment operator
  BPHFullSelect(const BPHFullSelect<T>& x);
  BPHFullSelect& operator=(const BPHFullSelect<T>& x);

  /** Destructor
   */
  ~BPHFullSelect() override = default;

  /** Operations
   */
  /// accept function
  bool accept(const typename T::AcceptArg& cand, const BPHRecoBuilder* build) const override {
    return Base::select(cand, build);
  }
};

template <class T = BPHFullSelect<BPHRecoSelect>>
class BPHMultiSelect : public T {
public:
  /** Constructor
   */
  BPHMultiSelect(BPHSelectOperation::mode op) : T(op) {}

  // deleted copy constructor and assignment operator
  BPHMultiSelect(const BPHMultiSelect<T>& x) = delete;
  BPHMultiSelect& operator=(const BPHMultiSelect<T>& x) = delete;

  /** Destructor
   */
  ~BPHMultiSelect() override = default;

  /** Operations
   */
  /// no override or new function, everything taken from base
};

#endif

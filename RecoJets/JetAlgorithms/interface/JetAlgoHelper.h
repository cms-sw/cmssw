#ifndef JetAlgorithms_JetAlgoHelper_h
#define JetAlgorithms_JetAlgoHelper_h

// Various simple tools
// F.Ratnikov, UMd
// $Id: JetAlgoHelper.h,v 1.4 2006/12/05 18:37:43 fedor Exp $

#include<limits>
#include <iostream>
#include <cmath>

#include "PhysicsTools/Utilities/interface/PtComparator.h"
#include "PhysicsTools/Utilities/interface/EtComparator.h"


template <class T>
class GreaterByPtRef {
 public:
  int operator()(const T& a1, const T& a2) {
    if (!a1) return 0;
    if (!a2) return 1;
    NumericSafeGreaterByPt <typename T::value_type> comp;
    return comp.operator () (*a1, *a2);
  }
};
template <class T>
class GreaterByPtPtr {
 public:
  int operator()(const T* a1, const T* a2) {
    if (!a1) return 0;
    if (!a2) return 1;
    NumericSafeGreaterByPt <T> comp;
    return comp.operator () (*a1, *a2);
  }
};

template <class T>
class GreaterByEtRef {
 public:
  int operator()(const T& a1, const T& a2) {
    if (!a1) return 0;
    if (!a2) return 1;
    NumericSafeGreaterByEt <typename T::value_type> comp;
    return comp.operator () (*a1, *a2);
  }
};

#endif

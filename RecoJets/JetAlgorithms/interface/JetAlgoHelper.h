#ifndef JetAlgorithms_JetAlgoHelper_h
#define JetAlgorithms_JetAlgoHelper_h

// Various simple tools
// F.Ratnikov, UMd
// $Id: JetAlgoHelper.h,v 1.2 2006/10/02 22:54:31 fedor Exp $

#include<limits>
#include <iostream>
#include <cmath>

template <class T>
class GreaterByPt {
 public:
  int operator()(const T& a1, const T& a2) {
    return
      abs (a1.pt()-a2.pt()) > std::numeric_limits<double>::epsilon() ? a1.pt() > a2.pt() :
      abs (a1.px()-a2.px()) > std::numeric_limits<double>::epsilon() ? a1.px() > a2.px() :
      abs (a1.py()-a2.py()) > std::numeric_limits<double>::epsilon() ? a1.py() > a2.py() :
      a1.pz() > a2.pz();
  }
  int operator()(const T* a1, const T* a2) {
    if (!a1) return 0;
    if (!a2) return 1;
    return operator () (*a1, *a2);
  }
};

template <class T>
class GreaterByEt {
 public:
  int operator()(const T& a1, const T& a2) {
    return
      abs (a1.et()-a2.et()) > std::numeric_limits<double>::epsilon() ? a1.et() > a2.et() :
      abs (a1.px()-a2.px()) > std::numeric_limits<double>::epsilon() ? a1.px() > a2.px() :
      abs (a1.py()-a2.py()) > std::numeric_limits<double>::epsilon() ? a1.py() > a2.py() :
      a1.pz() > a2.pz();
  }
  int operator()(const T* a1, const T* a2) {
    if (!a1) return 0;
    if (!a2) return 1;
    return operator () (*a1, *a2);
  }
};

template <class T> 
T deltaphi (T phi1, T phi2) { 
  T result = phi1 - phi2;
  while (result > M_PI) result -= 2*M_PI;
  while (result <= -M_PI) result += 2*M_PI;
  return result;
}

template <class T>
T deltaR2 (T eta1, T phi1, T eta2, T phi2) {
  T deta = eta1 - eta2;
  T dphi = deltaphi (phi1, phi2);
  return deta*deta + dphi*dphi;
}

template <class T>
T deltaR (T eta1, T phi1, T eta2, T phi2) {return sqrt (deltaR2 (eta1, phi1, eta2, phi2));}

#endif

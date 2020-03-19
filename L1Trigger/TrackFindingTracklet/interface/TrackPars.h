#ifndef TRACKPARS_H
#define TRACKPARS_H

#include <iostream>
#include <fstream>
#include <assert.h>
#include <math.h>
#include <vector>
#include <map>

#include "Util.h"

using namespace std;

template <class T>
class TrackPars{

public:

  TrackPars() {}
  
  TrackPars(T rinv, T phi0, T d0, T t, T z0) {
    rinv_=rinv;
    phi0_=phi0;
    d0_=d0;
    t_=t;
    z0_=z0;
  }

  ~TrackPars() {

  }
  
  void init(T rinv, T phi0, T d0, T t, T z0) {
    rinv_=rinv;
    phi0_=phi0;
    d0_=d0;
    t_=t;
    z0_=z0;
  }

  T rinv() const {return rinv_;}
  T phi0() const {return phi0_;}
  T d0() const {return d0_;}
  T t() const {return t_;}
  T z0() const {return z0_;}

  T& rinv() {return rinv_;}
  T& phi0() {return phi0_;}
  T& d0() {return d0_;}
  T& t() {return t_;}
  T& z0() {return z0_;}

  
private:
  
  T rinv_;
  T phi0_;
  T d0_;  
  T t_;
  T z0_;

};

#endif




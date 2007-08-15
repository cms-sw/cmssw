#ifndef PYQUEN_WRAPPER_H
#define PYQUEN_WRAPPER_H

/*
 *
 * Wrapper for FORTRAN version of PYQUEN
 *
 * Camelia Mironov
 *
 */
                                                       
// PYQUEN routine declaration

extern "C" {
  void pyquen_(float& a,int& ifb,float& bfix);
}
#define PYQUEN pyquen_ 

extern "C" {
  extern struct{
    float bgen;
  }plfpar_;
}
#define plfpar plfpar_


extern "C" {
  extern struct{
    float T0u;
    float tau0u;
    int   nfu;
    int   ienglu;
    int   ianglu;
  }pyqpar_;
}
#define pyqpar pyqpar_

#endif  // PYQUEN_WRAPPER_H

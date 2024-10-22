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
void pyquen_(double& a, int& ifb, double& bfix, double& bmin, double& bmax);
}
#define PYQUEN pyquen_

extern "C" {
void pyqver_(int&, int&, int&, int&);
}
#define PYQVER pyqver_

extern "C" {
extern struct {
  double bgen;
} plfpar_;
}
#define plfpar plfpar_

extern "C" {
extern struct {
  double T0u;
  double tau0u;
  int nfu;
  int ienglu;
  int ianglu;
} pyqpar_;
}
#define pyqpar pyqpar_

#endif  // PYQUEN_WRAPPER_H

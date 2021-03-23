#ifndef _CLEHP_2_SMATRIX_MIGRATION_H_
#define _CLEHP_2_SMATRIX_MIGRATION_H_

#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"

#include "CLHEP/Matrix/Matrix.h"
#include "CLHEP/Matrix/Vector.h"
#include "CLHEP/Matrix/SymMatrix.h"

#include <cstring>
#include <cassert>

/*
template <unsigned int N1, unsigned int N2> 
    ROOT::Math::SMatrix<double,N1,N2,ROOT::Math::RepStd<double,N1,N2> > clhep2smatrix(const CLHEP::HepMatrix &mt) {
        typedef ROOT::Math::SMatrix<double,N1,N2,ROOT::Math::RepStd<double,N1,N2> > RM;
        RM rm;
        memcpy(rm.Array(), &matmt(1,1), (N1*N2)*sizeof(double));
        return rm;
    }
template <unsigned int N1> 
    ROOT::Math::SMatrix<double,N1,N1,ROOT::Math::RepSym<double,N1> > clhep2smatrix(const CLHEP::HepSymMatrix &mt) {
        typedef ROOT::Math::SMatrix<double,N1,N1,ROOT::Math::RepSym<double,N1> > RM;
        RM rm;
        memcpy(rm.Array(), &matmt(1,1), (D1*D2)*sizeof(double));
        return rm;
    }
*/

template <unsigned int N1, unsigned int N2>
ROOT::Math::SMatrix<double, N1, N2, typename ROOT::Math::MatRepStd<double, N1, N2> > asSMatrix(
    const CLHEP::HepMatrix &m) {
  typedef typename ROOT::Math::MatRepStd<double, N1, N2> REP;
  assert(m.num_row() == N1);
  assert(m.num_col() == N2);
  return ROOT::Math::SMatrix<double, N1, N2, REP>(&m(1, 1), REP::kSize);
}

template <unsigned int N1>
ROOT::Math::SMatrix<double, N1, N1, typename ROOT::Math::MatRepSym<double, N1> > asSMatrix(
    const CLHEP::HepSymMatrix &m) {
  typedef typename ROOT::Math::MatRepSym<double, N1> REP;
  assert(m.num_row() == N1);
  return ROOT::Math::SMatrix<double, N1, N1, REP>(&m(1, 1), REP::kSize);
}

template <unsigned int N1>
ROOT::Math::SVector<double, N1> asSVector(const CLHEP::HepVector &m) {
  return ROOT::Math::SVector<double, N1>(&m[0], N1);
}

template <unsigned int N>
CLHEP::HepVector asHepVector(const ROOT::Math::SVector<double, N> &v) {
  CLHEP::HepVector hv(N);
  memcpy(&hv[0], &v[0], N * sizeof(double));
  return hv;
}

template <unsigned int N1, unsigned int N2>
CLHEP::HepMatrix asHepMatrix(
    const ROOT::Math::SMatrix<double, N1, N2, typename ROOT::Math::MatRepStd<double, N1, N2> > &rm) {
  CLHEP::HepMatrix am(N1, N2);
  memcpy(&am(1, 1), rm.Array(), N1 * N2 * sizeof(double));
  return am;
}

template <unsigned int N1>
CLHEP::HepSymMatrix asHepMatrix(
    const ROOT::Math::SMatrix<double, N1, N1, typename ROOT::Math::MatRepSym<double, N1> > &rm) {
  CLHEP::HepSymMatrix am(N1);
  memcpy(&am(1, 1), rm.Array(), (N1 * (N1 + 1)) / 2 * sizeof(double));
  return am;
}

#endif

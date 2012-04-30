#ifndef _CLEHP_2_SMATRIX_MIGRATION_H_
#define _CLEHP_2_SMATRIX_MIGRATION_H_

#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"
#include <cstring>

#ifdef  CLHEP_CMS_MODS // fast access to matrix in clhep

template<unsigned int N1, unsigned int N2> 
    ROOT::Math::SMatrix<double,N1,N2, typename ROOT::Math::MatRepStd<double,N1,N2> > asSMatrix(const CLHEP::HepMatrix &m) {
        typedef typename ROOT::Math::MatRepStd<double,N1,N2>  REP;
        assert(m.dataSize()==REP::kSize);
        return  ROOT::Math::SMatrix<double,N1,N2,REP> (m.data(), REP::kSize);
    }

template<unsigned int N1> 
    ROOT::Math::SMatrix<double,N1,N1, typename ROOT::Math::MatRepSym<double,N1> > asSMatrix(const CLHEP::HepSymMatrix &m) {
        typedef typename ROOT::Math::MatRepSym<double,N1>  REP;
        assert(m.dataSize()==REP::kSize);
        return  ROOT::Math::SMatrix<double,N1,N1,REP> (m.data(), REP::kSize);
    }

template<unsigned int N1> 
    ROOT::Math::SVector<double,N1> asSVector(const CLHEP::HepVector &m) {
        return  ROOT::Math::SVector<double,N1> (&m[0], N1);
    }

template<unsigned int N> CLHEP::HepVector asHepVector(const ROOT::Math::SVector<double,N> &v) {
    CLHEP::HepVector hv(N);
    memcpy(&hv[0], &v[0], N*sizeof(double));
    return hv;
    }

template<unsigned int N1, unsigned int N2> CLHEP::HepMatrix asHepMatrix (
     const ROOT::Math::SMatrix<double,N1,N2, typename ROOT::Math::MatRepStd<double,N1,N2> > &rm) {
        CLHEP::HepMatrix am(N1,N2);
        memcpy(am.data(), rm.Array(), N1*N2*sizeof(double));
        return am;
    }

template<unsigned int N1> CLHEP::HepSymMatrix asHepMatrix (
     const ROOT::Math::SMatrix<double,N1,N1, typename ROOT::Math::MatRepSym<double,N1> > &rm) {
        CLHEP::HepSymMatrix am(N1);
        memcpy(am.data(), rm.Array(), (N1*(N1+1))/2*sizeof(double));
        return am;
    }

#else
template<unsigned int N1, unsigned int N2> 
    ROOT::Math::SMatrix<double,N1,N2, typename ROOT::Math::MatRepStd<double,N1,N2> > asSMatrix(const CLHEP::HepMatrix &m) {
        typedef typename ROOT::Math::MatRepStd<double,N1,N2>  REP;
        assert(m.num_row() == N1); assert(m.num_col() == N2);
        return  ROOT::Math::SMatrix<double,N1,N2,REP> (&m(1,1), REP::kSize);
    }

template<unsigned int N1> 
    ROOT::Math::SMatrix<double,N1,N1, typename ROOT::Math::MatRepSym<double,N1> > asSMatrix(const CLHEP::HepSymMatrix &m) {
        typedef typename ROOT::Math::MatRepSym<double,N1>  REP;
        assert(m.num_row() == N1);
        return  ROOT::Math::SMatrix<double,N1,N1,REP> (&m(1,1), REP::kSize);
    }

template<unsigned int N1> 
    ROOT::Math::SVector<double,N1> asSVector(const CLHEP::HepVector &m) {
        return  ROOT::Math::SVector<double,N1> (&m[0], N1);
    }

template<unsigned int N> CLHEP::HepVector asHepVector(const ROOT::Math::SVector<double,N> &v) {
    CLHEP::HepVector hv(N);
    memcpy(&hv[0], &v[0], N*sizeof(double));
    return hv;
    }

template<unsigned int N1, unsigned int N2> CLHEP::HepMatrix asHepMatrix (
     const ROOT::Math::SMatrix<double,N1,N2, typename ROOT::Math::MatRepStd<double,N1,N2> > &rm) {
        CLHEP::HepMatrix am(N1,N2);
        memcpy(&am(1,1), rm.Array(), N1*N2*sizeof(double));
        return am;
    }

template<unsigned int N1> CLHEP::HepSymMatrix asHepMatrix (
     const ROOT::Math::SMatrix<double,N1,N1, typename ROOT::Math::MatRepSym<double,N1> > &rm) {
        CLHEP::HepSymMatrix am(N1);
        memcpy(&am(1,1), rm.Array(), (N1*(N1+1))/2*sizeof(double));
        return am;
    }

#endif

#endif

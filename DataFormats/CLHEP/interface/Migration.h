#ifndef _CLEHP_2_SMATRIX_MIGRATION_H_
#define _CLEHP_2_SMATRIX_MIGRATION_H_

#include <Math/SVector.h>
#include <Math/SMatrix.h>
#include <cstring>

// Use ".!" in VIM
// for I in $(seq 1 6); do echo "typedef ROOT::Math::SVector<double,$I> AlgebraicVector$I;"; done
typedef ROOT::Math::SVector<double,1> AlgebraicVector1;
typedef ROOT::Math::SVector<double,2> AlgebraicVector2;
typedef ROOT::Math::SVector<double,3> AlgebraicVector3;
typedef ROOT::Math::SVector<double,4> AlgebraicVector4;
typedef ROOT::Math::SVector<double,5> AlgebraicVector5;
typedef ROOT::Math::SVector<double,6> AlgebraicVector6;

// for I in $(seq 1 6); do echo "typedef ROOT::Math::SMatrix<double,$I,$I,ROOT::Math::MatRepSym<double,$I> > AlgebraicSymMatrix$I$I;"; done
typedef ROOT::Math::SMatrix<double,1,1,ROOT::Math::MatRepSym<double,1> > AlgebraicSymMatrix11;
typedef ROOT::Math::SMatrix<double,2,2,ROOT::Math::MatRepSym<double,2> > AlgebraicSymMatrix22;
typedef ROOT::Math::SMatrix<double,3,3,ROOT::Math::MatRepSym<double,3> > AlgebraicSymMatrix33;
typedef ROOT::Math::SMatrix<double,4,4,ROOT::Math::MatRepSym<double,4> > AlgebraicSymMatrix44;
typedef ROOT::Math::SMatrix<double,5,5,ROOT::Math::MatRepSym<double,5> > AlgebraicSymMatrix55;
typedef ROOT::Math::SMatrix<double,6,6,ROOT::Math::MatRepSym<double,6> > AlgebraicSymMatrix66;

// for I in $(seq 1 6); do for J in $(seq 1 6); do echo "typedef ROOT::Math::SMatrix<double,$I,$J,ROOT::Math::MatRepStd<double,$I,$J> > AlgebraicMatrix$I$J;"; done; done
typedef ROOT::Math::SMatrix<double,1,1,ROOT::Math::MatRepStd<double,1,1> > AlgebraicMatrix11;
typedef ROOT::Math::SMatrix<double,1,2,ROOT::Math::MatRepStd<double,1,2> > AlgebraicMatrix12;
typedef ROOT::Math::SMatrix<double,1,3,ROOT::Math::MatRepStd<double,1,3> > AlgebraicMatrix13;
typedef ROOT::Math::SMatrix<double,1,4,ROOT::Math::MatRepStd<double,1,4> > AlgebraicMatrix14;
typedef ROOT::Math::SMatrix<double,1,5,ROOT::Math::MatRepStd<double,1,5> > AlgebraicMatrix15;
typedef ROOT::Math::SMatrix<double,1,6,ROOT::Math::MatRepStd<double,1,6> > AlgebraicMatrix16;
typedef ROOT::Math::SMatrix<double,2,1,ROOT::Math::MatRepStd<double,2,1> > AlgebraicMatrix21;
typedef ROOT::Math::SMatrix<double,2,2,ROOT::Math::MatRepStd<double,2,2> > AlgebraicMatrix22;
typedef ROOT::Math::SMatrix<double,2,3,ROOT::Math::MatRepStd<double,2,3> > AlgebraicMatrix23;
typedef ROOT::Math::SMatrix<double,2,4,ROOT::Math::MatRepStd<double,2,4> > AlgebraicMatrix24;
typedef ROOT::Math::SMatrix<double,2,5,ROOT::Math::MatRepStd<double,2,5> > AlgebraicMatrix25;
typedef ROOT::Math::SMatrix<double,2,6,ROOT::Math::MatRepStd<double,2,6> > AlgebraicMatrix26;
typedef ROOT::Math::SMatrix<double,3,1,ROOT::Math::MatRepStd<double,3,1> > AlgebraicMatrix31;
typedef ROOT::Math::SMatrix<double,3,2,ROOT::Math::MatRepStd<double,3,2> > AlgebraicMatrix32;
typedef ROOT::Math::SMatrix<double,3,3,ROOT::Math::MatRepStd<double,3,3> > AlgebraicMatrix33;
typedef ROOT::Math::SMatrix<double,3,4,ROOT::Math::MatRepStd<double,3,4> > AlgebraicMatrix34;
typedef ROOT::Math::SMatrix<double,3,5,ROOT::Math::MatRepStd<double,3,5> > AlgebraicMatrix35;
typedef ROOT::Math::SMatrix<double,3,6,ROOT::Math::MatRepStd<double,3,6> > AlgebraicMatrix36;
typedef ROOT::Math::SMatrix<double,4,1,ROOT::Math::MatRepStd<double,4,1> > AlgebraicMatrix41;
typedef ROOT::Math::SMatrix<double,4,2,ROOT::Math::MatRepStd<double,4,2> > AlgebraicMatrix42;
typedef ROOT::Math::SMatrix<double,4,3,ROOT::Math::MatRepStd<double,4,3> > AlgebraicMatrix43;
typedef ROOT::Math::SMatrix<double,4,4,ROOT::Math::MatRepStd<double,4,4> > AlgebraicMatrix44;
typedef ROOT::Math::SMatrix<double,4,5,ROOT::Math::MatRepStd<double,4,5> > AlgebraicMatrix45;
typedef ROOT::Math::SMatrix<double,4,6,ROOT::Math::MatRepStd<double,4,6> > AlgebraicMatrix46;
typedef ROOT::Math::SMatrix<double,5,1,ROOT::Math::MatRepStd<double,5,1> > AlgebraicMatrix51;
typedef ROOT::Math::SMatrix<double,5,2,ROOT::Math::MatRepStd<double,5,2> > AlgebraicMatrix52;
typedef ROOT::Math::SMatrix<double,5,3,ROOT::Math::MatRepStd<double,5,3> > AlgebraicMatrix53;
typedef ROOT::Math::SMatrix<double,5,4,ROOT::Math::MatRepStd<double,5,4> > AlgebraicMatrix54;
typedef ROOT::Math::SMatrix<double,5,5,ROOT::Math::MatRepStd<double,5,5> > AlgebraicMatrix55;
typedef ROOT::Math::SMatrix<double,5,6,ROOT::Math::MatRepStd<double,5,6> > AlgebraicMatrix56;
typedef ROOT::Math::SMatrix<double,6,1,ROOT::Math::MatRepStd<double,6,1> > AlgebraicMatrix61;
typedef ROOT::Math::SMatrix<double,6,2,ROOT::Math::MatRepStd<double,6,2> > AlgebraicMatrix62;
typedef ROOT::Math::SMatrix<double,6,3,ROOT::Math::MatRepStd<double,6,3> > AlgebraicMatrix63;
typedef ROOT::Math::SMatrix<double,6,4,ROOT::Math::MatRepStd<double,6,4> > AlgebraicMatrix64;
typedef ROOT::Math::SMatrix<double,6,5,ROOT::Math::MatRepStd<double,6,5> > AlgebraicMatrix65;
typedef ROOT::Math::SMatrix<double,6,6,ROOT::Math::MatRepStd<double,6,6> > AlgebraicMatrix66;


/// ============= When we need templated root objects 
template <unsigned int D1, unsigned int D2=D1> struct AlgebraicROOTObject {
    typedef typename ROOT::Math::SVector<double,D1> Vector;
    typedef typename ROOT::Math::SMatrix<double,D1,D1,ROOT::Math::MatRepSym<double,D1> > SymMatrix;
    typedef typename ROOT::Math::SMatrix<double,D1,D2,ROOT::Math::MatRepStd<double,D1,D2> > Matrix;
};

typedef ROOT::Math::SMatrixIdentity AlgebraicMatrixID;


/*
template <unsigned int N1, unsigned int N2> 
    ROOT::Math::SMatrix<double,N1,N2,ROOT::Math::RepStd<double,N1,N2> > clhep2smatrix(const HepMatrix &mt) {
        typedef ROOT::Math::SMatrix<double,N1,N2,ROOT::Math::RepStd<double,N1,N2> > RM;
        RM rm;
        memcpy(rm.Array(), &matmt(1,1), (N1*N2)*sizeof(double));
        return rm;
    }
template <unsigned int N1> 
    ROOT::Math::SMatrix<double,N1,N1,ROOT::Math::RepSym<double,N1> > clhep2smatrix(const HepSymMatrix &mt) {
        typedef ROOT::Math::SMatrix<double,N1,N1,ROOT::Math::RepSym<double,N1> > RM;
        RM rm;
        memcpy(rm.Array(), &matmt(1,1), (D1*D2)*sizeof(double));
        return rm;
    }
*/

template<unsigned int N1, unsigned int N2> 
    ROOT::Math::SMatrix<double,N1,N2, typename ROOT::Math::MatRepStd<double,N1,N2> > asSMatrix(const HepMatrix &m) {
        typedef typename ROOT::Math::MatRepStd<double,N1,N2>  REP;
        assert(m.num_row() == N1); assert(m.num_col() == N2);
        return  ROOT::Math::SMatrix<double,N1,N2,REP> (&m(1,1), REP::kSize);
    }

template<unsigned int N1> 
    ROOT::Math::SMatrix<double,N1,N1, typename ROOT::Math::MatRepSym<double,N1> > asSMatrix(const HepSymMatrix &m) {
        typedef typename ROOT::Math::MatRepSym<double,N1>  REP;
        assert(m.num_row() == N1);
        return  ROOT::Math::SMatrix<double,N1,N1,REP> (&m(1,1), REP::kSize);
    }

template<unsigned int N1> 
    ROOT::Math::SVector<double,N1> asSVector(const HepVector &m) {
        return  ROOT::Math::SVector<double,N1> (&m[0], N1);
    }

template<unsigned int N> HepVector asHepVector(const ROOT::Math::SVector<double,N> &v) {
    HepVector hv(N);
    memcpy(&hv[0], &v[0], N*sizeof(double));
    return hv;
    }

template<unsigned int N1, unsigned int N2> HepMatrix asHepMatrix (
     const ROOT::Math::SMatrix<double,N1,N2, typename ROOT::Math::MatRepStd<double,N1,N2> > &rm) {
        HepMatrix am(N1,N2);
        memcpy(&am(1,1), rm.Array(), N1*N2*sizeof(double));
        return am;
    }

template<unsigned int N1> HepSymMatrix asHepMatrix (
     const ROOT::Math::SMatrix<double,N1,N1, typename ROOT::Math::MatRepSym<double,N1> > &rm) {
        HepSymMatrix am(N1);
        memcpy(&am(1,1), rm.Array(), (N1*(N1+1))/2*sizeof(double));
        return am;
    }


#endif

// #include "DataFormats/Math/interface/MulSymMatrix.h"
#include "Math/SMatrix.h"

// typedef unsigned int IndexType;
typedef unsigned long long IndexType;
template<typename T, IndexType N>
inline void
mult(ROOT::Math::SMatrix<T,N,N,ROOT::Math::MatRepSym<T,N> > & a, 
     ROOT::Math::SMatrix<T,N,N,ROOT::Math::MatRepSym<T,N> > const & rh,
     ROOT::Math::SMatrix<T,N,N,ROOT::Math::MatRepSym<T,N> > const & lh) {
  // a(i,j) = r(i,k)*l(k,j)
    for (IndexType i=0; i!=N; ++i) {
      IndexType off_i = a.fRep.Offsets()(i,0);
  for (IndexType k=0; k!=N; ++k) {
    IndexType off_k = a.fRep.Offsets()(k,0);
      if (k<i) {
	for (IndexType j=0; j!=(k+1); ++j) 
	  a.Array()[off_i+j] += rh(i,k)*lh.Array()[off_k+j];
	for (IndexType j=k+1; j!=(i+1); ++j) 
	  a.Array()[off_i+j] += rh(i,k)*lh(k,j);
      }
      else
	for (IndexType j=0; j!=(i+1); ++j) 
	  a.Array()[off_i+j] += rh(i,k)*lh.Array()[off_k+j];
    }
  }
}

#include<iostream>
#include "FWCore/Utilities/interface/HRRealTime.h"

#include<random>

namespace {
  std::mt19937 eng;
  std::uniform_real_distribution<double> rgen(-5.,5.);
  template<typename T, IndexType N>
  inline void
  fillRandom(ROOT::Math::SMatrix<T,N,N,ROOT::Math::MatRepSym<T,N> > & a) {
    for (IndexType k=0; k!=N; ++k)
      for (IndexType j=0; j!=(k+1); ++j)
	a(k,j) =  rgen(eng);
  }
}

template<typename T, IndexType N>
void go( edm::HRTimeType & s1, edm::HRTimeType & s2,  edm::HRTimeType & s3,  edm::HRTimeType & s4, bool print) {
  typedef ROOT::Math::SMatrix<T,N,N,ROOT::Math::MatRepStd<T,N> > Matrix;
  typedef ROOT::Math::SMatrix<T,N,N,ROOT::Math::MatRepSym<T,N> > SymMatrix;


  SymMatrix rh;  fillRandom(rh);
  SymMatrix lh;  fillRandom(lh);

  Matrix mrh = rh;
  Matrix mlh = lh;

  SymMatrix a; 
  s1 = edm::hrRealTime();
  mult(a,rh,lh);
  s1 =  edm::hrRealTime() -s1;

  SymMatrix b;
  s2 = edm::hrRealTime();
  ROOT::Math::AssignSym::Evaluate(b, rh*lh);
  s2 =  edm::hrRealTime() -s2;


  Matrix m;
  s3 = edm::hrRealTime();
  m = mrh*mlh;
  s3 =  edm::hrRealTime() -s3;

  SymMatrix sm;
  s4 = edm::hrRealTime();
  ROOT::Math::AssignSym::Evaluate(sm,mrh*mlh);
  s4 =  edm::hrRealTime() -s4;

  SymMatrix smm; ROOT::Math::AssignSym::Evaluate(smm,m);

  if (print) {
    if (smm!=b) std::cout << "problem with SMatrix *" << std::endl;
    if (sm!=b) std::cout << "problem with SMatrix evaluate" << std::endl;
    if (a!=b) std::cout << "problem with MulSymMatrix" << std::endl;
    if (a!=smm) std::cout << "problem with MulSymMatrix twice" << std::endl;
    
    std::cout << "sym mult   " << s1 << std::endl;
    std::cout << "sym    *   " << s2 << std::endl;
    std::cout << "mat    *   " << s3 << std::endl;
    std::cout << "mat as sym " << s4 << std::endl;
  }

}
 
int main() {
  edm::HRTimeType s1=0, s2=0, s3=0, s4=0;
  go<double,15>(s1,s2,s3,s4,true);

  edm::HRTimeType t1=0; edm::HRTimeType t2=0;  edm::HRTimeType t3=0; edm::HRTimeType t4=0;
  for (int  i=0; i!=50000; ++i) {
    go<double,15>(s1,s2,s3,s4, false);
    t1+=s1; t2+=s2; t3+=s3; t4+=s4;
  }
  std::cout << "sym mult   " << t1/50000 << std::endl;
  std::cout << "sym    *   " << t2/50000 << std::endl;
  std::cout << "mat    *   " << t3/50000 << std::endl;
  std::cout << "mat as sym " << t4/50000 << std::endl;

  return 0;
}

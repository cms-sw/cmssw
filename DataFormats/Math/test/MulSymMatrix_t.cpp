// #include "DataFormats/Math/interface/MulSymMatrix.h"
#include "Math/SMatrix.h"

typedef unsigned int IndexType;
//typedef unsigned long long IndexType;
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


template<typename T, IndexType N>
inline void
mult(ROOT::Math::SMatrix<T,N,N,ROOT::Math::MatRepStd<T,N> > & a, 
     ROOT::Math::SMatrix<T,N,N,ROOT::Math::MatRepStd<T,N> > const & rh,
     ROOT::Math::SMatrix<T,N,N,ROOT::Math::MatRepStd<T,N> > const & lh) {
  // a(i,j) = r(i,k)*l(k,j)
  for (IndexType i=0; i!=N; ++i) {
    for (IndexType j=0; j<=i; ++j) {
      // a(i,j)=0;
	//	a(i,j) += rh(i,k)*lh(k,j);
      for (IndexType k=0; k!=N; ++k) 
	a(i,j) += rh(i,k)*lh(j,k);
    }
    
  }
  
  for (IndexType i=0; i!=N-1; ++i) 
    for (IndexType j=i+1; j!=N; ++j)
      a(i,j)=a(j,i);
 
}

// U(i,k) * A(k,l) * U(j,l)
template<typename T, IndexType N>
inline void
similarity(ROOT::Math::SMatrix<T,N,N,ROOT::Math::MatRepStd<T,N> > & b, 
	   ROOT::Math::SMatrix<T,N,N,ROOT::Math::MatRepStd<T,N> > const & u,
	   ROOT::Math::SMatrix<T,N,N,ROOT::Math::MatRepStd<T,N> > const & a) {
  T s[N];
  for (IndexType i=0; i!=N; ++i) 
    for (IndexType j=0; j<=i; ++j) { 
      for (IndexType k=0; k!=N; ++k) {
	s[k]=0;
	for (IndexType l=0; l!=N; ++l) 
	  s[k] += a(k,l)*u(j,l);
      }
      for (IndexType k=0; k!=N; ++k)
	b(i,j) += u(i,k)*s[k];
    }
}

// U(k,i) * A(k,l) * U(l,j)
template<typename T, IndexType N>
inline void
similarityT(ROOT::Math::SMatrix<T,N,N,ROOT::Math::MatRepStd<T,N> > & b, 
	    ROOT::Math::SMatrix<T,N,N,ROOT::Math::MatRepStd<T,N> > const & u,
	    ROOT::Math::SMatrix<T,N,N,ROOT::Math::MatRepStd<T,N> > const & a) {
  for (IndexType i=0; i!=N; ++i) 
    for (IndexType j=0; j<=i; ++j) 
      for (IndexType k=0; k!=N; ++k) 
	for (IndexType l=0; l!=N; ++l) 
	  b(i,j) += u(k,i)*a(k,l)*u(l,j);
}

template<typename M1, typename M2>
double eps(M1 const & m1, M2 const & m2) {
  IndexType N = M1::kRows;
  double ret=0.;
  for (IndexType i=0; i!=N; ++i)
    for (IndexType j=0; j!=N; ++j) 
      ret = std::max(ret,std::abs(m1(i,j)-m2(i,j)));
  return ret;
}



#include<iostream>
#include "FWCore/Utilities/interface/HRRealTime.h"

#include<random>

namespace {
  std::mt19937 eng;
  std::uniform_real_distribution<double> rgen(-5.,5.);
  template<typename T, IndexType N>
  inline void
  fillRandom(ROOT::Math::SMatrix<T,N,N,ROOT::Math::MatRepStd<T,N> > & a) {
    for (IndexType k=0; k!=N; ++k)
      for (IndexType j=0; j!=N; ++j)
	a(k,j) =  rgen(eng);
  }
}

template<typename T, IndexType N>
void go( edm::HRTimeType & s1, edm::HRTimeType & s2,  
	 edm::HRTimeType & s3,  edm::HRTimeType & s4,  
	 edm::HRTimeType & s5,
	 edm::HRTimeType & s6,  edm::HRTimeType & s7,
	 bool print) {
  typedef ROOT::Math::SMatrix<T,N,N,ROOT::Math::MatRepStd<T,N> > Matrix;
  typedef ROOT::Math::SMatrix<T,N,N,ROOT::Math::MatRepSym<T,N> > SymMatrix;

  ROOT::Math::SMatrixIdentity id;

  SymMatrix is(id);
  Matrix im(id);

  Matrix j1;  fillRandom(j1);
  Matrix j2;  fillRandom(j2);

  SymMatrix rh;
  s6 = edm::hrRealTime();
  rh =  ROOT::Math::Similarity(j1, is);
  s6 =  edm::hrRealTime() -s6;

  Matrix mlh;
  s7 = edm::hrRealTime();
  similarity(mlh,j2,im);
  s7 =  edm::hrRealTime() -s7;

  SymMatrix lh = ROOT::Math::Similarity(j2, is);

  Matrix mrh = rh;




  SymMatrix a; 
  s1 = edm::hrRealTime();
  mult(a,rh,lh);
  s1 =  edm::hrRealTime() -s1;

  SymMatrix b;
  s2 = edm::hrRealTime();
  ROOT::Math::AssignSym::Evaluate(b, rh*lh);
  s2 =  edm::hrRealTime() -s2;

  Matrix m0 = b;


  Matrix m;
  s3 = edm::hrRealTime();
  m = mrh*mlh;
  s3 =  edm::hrRealTime() -s3;

  SymMatrix sm;
  s4 = edm::hrRealTime();
  ROOT::Math::AssignSym::Evaluate(sm,mrh*mlh);
  s4 =  edm::hrRealTime() -s4;

  Matrix m2;
  s5 = edm::hrRealTime();
  mult(m2,mrh,mlh);
  s5 =  edm::hrRealTime() -s5;


  SymMatrix smm; ROOT::Math::AssignSym::Evaluate(smm,m);
  SymMatrix smm2; ROOT::Math::AssignSym::Evaluate(smm2,m2);

  if (print) {
    std::cout << "eps as   " << eps(rh,mrh) << std::endl;
    std::cout << "eps sim  " << eps(lh,mlh) << std::endl;
    std::cout << "eps s m  " << eps(m,b) << std::endl;
    std::cout << "eps s sm " << eps(m2,b) << std::endl;

    // std::cout << b << std::endl;
    // std::cout << m << std::endl;

    if (m!=m0) std::cout << "problem with SMatrix Assign " << eps(m,m0) << std::endl;
    if (smm!=b) std::cout << "problem with SMatrix * " << eps(smm,b) << std::endl;
    if (sm!=b) std::cout << "problem with SMatrix evaluate " << eps(sm,b) << std::endl;
    if (a!=b) std::cout << "problem with MulSymMatrix " << eps(a,b) << std::endl;
    if (a!=smm) std::cout << "problem with MulSymMatrix twice " << eps(a,smm) << std::endl;
    if (m!=m2) std::cout << "problem with MulMatrix " << eps(m,m2) << std::endl;
    if (smm!=smm2) std::cout << "problem with MulMatrix twice " << eps(smm,smm2) << std::endl;
    
    std::cout << "sym mult   " << s1 << std::endl;
    std::cout << "sym    *   " << s2 << std::endl;
    std::cout << "mat    *   " << s3 << std::endl;
    std::cout << "mat as sym " << s4 << std::endl;
    std::cout << "mat mult   " << s5 << std::endl;
    std::cout << "sym  sim   " << s6 << std::endl;
    std::cout << "loop sim   " << s7 << std::endl;
  
  }

}
 
int main() {
  edm::HRTimeType s1=0, s2=0, s3=0, s4=0, s5=0, s6=0, s7=0;
  go<double,3>(s1,s2,s3,s4,s5,s6,s7, true);
 

 go<double,15>(s1,s2,s3,s4,s5,s6,s7, true);

  edm::HRTimeType t1=0; edm::HRTimeType t2=0;  edm::HRTimeType t3=0; edm::HRTimeType t4=0; edm::HRTimeType t5=0;
  edm::HRTimeType t6=0; edm::HRTimeType t7=0;

  for (int  i=0; i!=50000; ++i) {
    go<double,15>(s1,s2,s3,s4,s5,s6,s7, false);
    t1+=s1; t2+=s2; t3+=s3; t4+=s4; t5+=s5;  t6+=s6; t7+=s7;
  }
  std::cout << "sym mult   " << t1/50000 << std::endl;
  std::cout << "sym    *   " << t2/50000 << std::endl;
  std::cout << "mat    *   " << t3/50000 << std::endl;
  std::cout << "mat as sym " << t4/50000 << std::endl;
  std::cout << "mat mult   " << t5/50000 << std::endl;
    std::cout << "sym  sim " << t6/50000 << std::endl;
    std::cout << "loop sim " << t7/50000 << std::endl;
 
  return 0;
}

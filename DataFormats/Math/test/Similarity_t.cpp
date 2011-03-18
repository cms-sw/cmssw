#include "Math/SMatrix.h"

typedef unsigned int IndexType;
//typedef unsigned long long IndexType;


namespace ROOT { 

  namespace Math { 

   /** 
       Force Expression evaluation from general to symmetric. 
       To be used when is known (like in similarity products) that the result 
       is symmetric
       Note this is function used in the simmilarity product: no check for temporary is 
       done since in that case is not needed
   */
   struct AssignAsSym
   {
      /// assign a symmetric matrix from an expression
      template <class T, 
		IndexType D,
                class A, 
                class R>
      static void Evaluate(SMatrix<T,D,D,MatRepStd<T,D> >& lhs,  const Expr<A,T,D,D,R>& rhs) 
     {
       //for(unsigned int i=0; i<D1*D2; ++i) lhs.fRep[i] = rhs.apply(i);
       for( IndexType i=0; i<D; ++i)
	 // storage of symmetric matrix is in lower block
	 for( IndexType j=0; j<=i; ++j) { 
	   lhs(i,j) = rhs(i,j);
	 }
       for (IndexType i=0; i!=D-1; ++i) 
	 for (IndexType j=i+1; j!=D; ++j)
	   lhs(i,j) = lhs(j,i);
     }
      /// assign the symmetric matric  from a general matrix  
     template <class T, 
	       IndexType D,
	       class R>
     static void Evaluate(SMatrix<T,D,D,MatRepStd<T,D> >& lhs,  const SMatrix<T,D,D,R>& rhs) 
     {
       //for(unsigned int i=0; i<D1*D2; ++i) lhs.fRep[i] = rhs.apply(i);
       for( IndexType i=0; i<D; ++i)
	 // storage of symmetric matrix is in lower block
	 for( IndexType j=0; j<=i; ++j) { 
	   lhs(i,j) = rhs(i,j);
	 }
       for (IndexType i=0; i!=D-1; ++i) 
	 for (IndexType j=i+1; j!=D; ++j)
	   lhs(i,j) = lhs(j,i);
     } 
   }; // struct AssignAsSym 
    
    
    template <class T,  IndexType D1,  IndexType D2, class R>
    inline SMatrix<T,D1,D1,MatRepSym<T,D1> > Similarity1(const SMatrix<T,D1,D2,R>& lhs, 
							 const SMatrix<T,D2,D2,MatRepStd<T,D2> >& rhs) {
      SMatrix<T,D1,D2, MatRepStd<T,D1,D2> > tmp = lhs * rhs;
      typedef  SMatrix<T,D1,D1,MatRepSym<T,D1> > SMatrixSym; 
      SMatrixSym mret; 
      AssignSym::Evaluate(mret,  tmp * Transpose(lhs)  ); 
      return mret; 
    }
    
    template <class T,  IndexType D1,  IndexType D2>
    inline SMatrix<T,D1,D1,MatRepStd<T,D1> > Similarity2(const SMatrix<T,D1,D2,MatRepStd<T,D1,D2> > & lhs, 
							 const SMatrix<T,D2,D2,MatRepStd<T,D2> > & rhs) {
      SMatrix<T,D1,D2, MatRepStd<T,D1,D2> > tmp = lhs * rhs;
      typedef  SMatrix<T,D1,D1,MatRepStd<T,D1> > SMatrixSym; 
      SMatrixSym mret; 
      AssignAsSym::Evaluate(mret,  tmp * Transpose(lhs)  ); 
      return mret; 
    }
  }
}

// U(i,k) * A(k,l) * U(j,l)
template<typename T, IndexType D1, IndexType D2>
inline void
similarity(ROOT::Math::SMatrix<T,D1,D1,ROOT::Math::MatRepStd<T,D1> > & b, 
	   ROOT::Math::SMatrix<T,D1,D2,ROOT::Math::MatRepStd<T,D1,D2> > const & u,
	   ROOT::Math::SMatrix<T,D2,D2,ROOT::Math::MatRepStd<T,D2> > const & a) {
  for (IndexType i=0; i!=D1; ++i) 
    for (IndexType j=0; j<=i; ++j) 
      for (IndexType k=0; k!=D2; ++k) 
	for (IndexType l=0; l!=D2; ++l) 
	  b(i,j) += u(i,k)*a(k,l)*u(j,l);
  /*
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
  */

 for (IndexType i=0; i!=D1-1; ++i) 
    for (IndexType j=i+1; j!=D1; ++j)
      b(i,j)=b(j,i);
 
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

template<typename M>
bool isSym(M const & m) {
  IndexType N = M::kRows;
  for (IndexType i=0; i!=N; ++i) 
    for (IndexType j=0; j<=i; ++j)
      if (m(i,j)!=m(j,i)) return false;
  return true;
}

#include<iostream>
#include "FWCore/Utilities/interface/HRRealTime.h"

#include<random>

namespace {
  std::mt19937 eng;
  std::uniform_real_distribution<double> rgen(-5.,5.);
  template<typename T,  IndexType D1,  IndexType D2 >
  inline void
  fillRandom(ROOT::Math::SMatrix<T,D1,D2,ROOT::Math::MatRepStd<T,D1,D2> > & a) {
    for (IndexType i=0; i!=D1; ++i) {
      for (IndexType j=0; j!=D2; ++j)
	a(i,j) =  rgen(eng);
    }
  }
  template<typename T,  IndexType N>
  inline void
  fillRandomSym(ROOT::Math::SMatrix<T,N,N,ROOT::Math::MatRepStd<T,N> > & a) {
    for (IndexType i=0; i!=N; ++i) {
      for (IndexType j=0; j<=i; ++j)
	a(i,j) =  rgen(eng);
    }
    for (IndexType i=0; i!=N-1; ++i) 
      for (IndexType j=i+1; j!=N; ++j)
	a(i,j)=a(j,i);
  }
}



template<typename T, IndexType D1,  IndexType D2 >
void go( edm::HRTimeType & s1, edm::HRTimeType & s2,  
	 edm::HRTimeType & s3,  edm::HRTimeType & s4, bool print) {

  typedef ROOT::Math::SMatrix<T,D1,D2,ROOT::Math::MatRepStd<T,D1,D2> > JMatrix;
  typedef ROOT::Math::SMatrix<T,D1,D1,ROOT::Math::MatRepStd<T,D1,D1> > Matrix1;
  typedef ROOT::Math::SMatrix<T,D1,D1,ROOT::Math::MatRepSym<T,D1> > SymMatrix1;
  typedef ROOT::Math::SMatrix<T,D2,D2,ROOT::Math::MatRepStd<T,D2,D2> > Matrix2;
  typedef ROOT::Math::SMatrix<T,D2,D2,ROOT::Math::MatRepSym<T,D2> > SymMatrix2;


  JMatrix lh; fillRandom(lh);
  Matrix2 rh; fillRandomSym(rh);
  
  SymMatrix2 srh; ROOT::Math::AssignSym::Evaluate(srh,rh);



  SymMatrix1 res1;
  s1 = edm::hrRealTime();
  res1 =  ROOT::Math::Similarity(lh, srh);
  s1 =  edm::hrRealTime() -s1;

  SymMatrix1 res2;
  s2 = edm::hrRealTime();
  res2 =  ROOT::Math::Similarity1(lh, rh);
  s2 =  edm::hrRealTime() -s2;

  Matrix1 res3;
  s3 = edm::hrRealTime();
  res3 =  ROOT::Math::Similarity2(lh, rh);
  s3 =  edm::hrRealTime() -s3;


  Matrix1 res4;
  s4 = edm::hrRealTime();
  similarity(res4,lh,rh);
  s4 =  edm::hrRealTime() -s4;

  if (print) {
    if (!isSym(rh))  std::cout << " rh is not sym" << std::endl;
    if (!isSym(res1))  std::cout << " res1 is not sym" << std::endl;
    if (!isSym(res2))  std::cout << " res2 is not sym" << std::endl;
    if (!isSym(res3))  std::cout << " res3 is not sym" << std::endl;
    if (!isSym(res4))  std::cout << " res4 is not sym" << std::endl;

    std::cout << D1 << "x"<< D2 << std::endl;
    std::cout << "eps sim  " << eps(res1,res2) << std::endl;
    std::cout << "eps std  " << eps(res1,res3) << std::endl;
    std::cout << "eps loop  " << eps(res1,res4) << std::endl;
  }

}



template<typename T, IndexType D1,  IndexType D2 >
void loop() {
  edm::HRTimeType t1=0; edm::HRTimeType t2=0;  edm::HRTimeType t3=0; edm::HRTimeType t4=0;
  edm::HRTimeType s1=0, s2=0, s3=0, s4=0;
  for (int  i=0; i!=50000; ++i) {
    go<T,D1,D2>(s1,s2,s3,s4, false);
    t1+=s1; t2+=s2; t3+=s3; t4+=s4;
  }

  std::cout << D1 << "x"<< D2 << std::endl;
  std::cout << "root sim " << t1/50000 << std::endl;
  std::cout << "sym  sim " << t2/50000 << std::endl;
  std::cout << "std  sim " << t3/50000 << std::endl;
  std::cout << "loop sim " << t4/50000 << std::endl;

}


int main() {

  edm::HRTimeType s1=0, s2=0, s3=0, s4=0;

  go<double,3,3>(s1,s2,s3,s4, true);
  go<double,5,5>(s1,s2,s3,s4, true);
  go<double,5,15>(s1,s2,s3,s4, true);
  go<double,15,15>(s1,s2,s3,s4, true);
  std::cout << std::endl;

  loop<double,3,3>();
  loop<double,5,5>();
  loop<double,5,15>();
  loop<double,15,15>();

  return 0;

}

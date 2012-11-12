// #include "DataFormats/Math/interface/MulSymMatrix.h"
#include "Math/SMatrix.h"

typedef unsigned int IndexType;
//typedef unsigned long long IndexType;


namespace ROOT { 

  namespace Math { 

   constexpr int fOff15x15[] = 
      { 0,1,3,6,10,15,21,28,36,45,55,66,78,91,105,1,2,4,7,11,16,22,29,37,46,56,67,79,92,106,3,4,5,8,12,17,23,30,38,47,57,68,80,93,107,6,7,8,9,13,18,24,31,39,48,58,69,81,94,108,10,11,12,13,14,19,25,32,40,49,59,70,82,95,109,15,16,17,18,19,20,26,33,41,50,60,71,83,96,110,21,22,23,24,25,26,27,34,42,51,61,72,84,97,111,28,29,30,31,32,33,34,35,43,52,62,73,85,98,112,36,37,38,39,40,41,42,43,44,53,63,74,86,99,113,45,46,47,48,49,50,51,52,53,54,64,75,87,100,114,55,56,57,58,59,60,61,62,63,64,65,76,88,101,115,66,67,68,69,70,71,72,73,74,75,76,77,89,102,116,78,79,80,81,82,83,84,85,86,87,88,89,90,103,117,91,92,93,94,95,96,97,98,99,100,101,102,103,104,118,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119 };
    
    constexpr int const * pfOff15x15[]= {fOff15x15+0,fOff15x15+15,fOff15x15+2*15,fOff15x15+3*15,fOff15x15+4*15,fOff15x15+5*15,fOff15x15+6*15,fOff15x15+7*15,fOff15x15+8*15,fOff15x15+9*15,fOff15x15+10*15,fOff15x15+11*15,fOff15x15+12*15,fOff15x15+13*15,fOff15x15+14*15};
    template<>
    struct RowOffsets<15> {
      RowOffsets() {}
      int operator()(unsigned int i, unsigned int j) const { return pfOff15x15[i][j]; } // fOff15x15[i*15+j]; }
      int apply(unsigned int i) const { return fOff15x15[i]; }
    };
  
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
      } 
   }; // struct AssignSym 
    

    template <class T,  IndexType D1,  IndexType D2, class R>
    inline SMatrix<T,D1,D1,MatRepSym<T,D1> > Similarity(const SMatrix<T,D1,D2,R>& lhs, const SMatrix<T,D2,D2,MatRepStd<T,D2> >& rhs) {
      SMatrix<T,D1,D2, MatRepStd<T,D1,D2> > tmp = lhs * rhs;
      typedef  SMatrix<T,D1,D1,MatRepSym<T,D1> > SMatrixSym; 
      SMatrixSym mret; 
      AssignSym::Evaluate(mret,  tmp * Transpose(lhs)  ); 
      return mret; 
    }

  }
}

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
  for (IndexType i=0; i!=N; ++i) 
    for (IndexType j=0; j<=i; ++j) 
      for (IndexType k=0; k!=N; ++k) 
	for (IndexType l=0; l!=N; ++l) 
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

 for (IndexType i=0; i!=N-1; ++i) 
    for (IndexType j=i+1; j!=N; ++j)
      b(i,j)=b(j,i);
 
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

  for (IndexType i=0; i!=N-1; ++i) 
    for (IndexType j=i+1; j!=N; ++j)
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
  template<typename T, IndexType N>
  inline void
  fillRandom(ROOT::Math::SMatrix<T,N,N,ROOT::Math::MatRepStd<T,N> > & a) {
    for (IndexType k=0; k!=N; ++k) {
      a(k,k) =  std::abs(rgen(eng));
      for (IndexType j=0; j<k; ++j)
	a(k,j) =  rgen(eng);
    }
    for (IndexType i=0; i!=N-1; ++i) 
      for (IndexType j=i+1; j!=N; ++j)
	a(i,j) = - a(j,i);
  }
}

template<typename T, IndexType N>
void go( edm::HRTimeType & s1, edm::HRTimeType & s2,  
	 edm::HRTimeType & s3,  edm::HRTimeType & s4,  
	 edm::HRTimeType & s5,
	 edm::HRTimeType & s6,  edm::HRTimeType & s7,  edm::HRTimeType & s8,
	 bool print) {
  typedef ROOT::Math::SMatrix<T,N,N,ROOT::Math::MatRepStd<T,N> > Matrix;
  typedef ROOT::Math::SMatrix<T,N,N,ROOT::Math::MatRepSym<T,N> > SymMatrix;

  ROOT::Math::SMatrixIdentity id;

  Matrix im1(id);
  Matrix im2(id);
  /*
  fillRandom(im1);
  im1 = im1*ROOT::Math::Transpose(im1);
  im1 *=im1;

  Matrix im2; fillRandom(im2);
  im2 = im2*ROOT::Math::Transpose(im2);
  im2 *=im2;
  */

  SymMatrix is1; ROOT::Math::AssignSym::Evaluate(is1, im1);
  SymMatrix is2; ROOT::Math::AssignSym::Evaluate(is2, im2);

  Matrix j1;  fillRandom(j1);
  Matrix j2;  fillRandom(j2);

  SymMatrix rh;
  s6 = edm::hrRealTime();
  rh =  ROOT::Math::Similarity(j1, is1);
  s6 =  edm::hrRealTime() -s6;

  SymMatrix rh2;
  s7 = edm::hrRealTime();
  rh2 =  ROOT::Math::Similarity(j1, im1);
  s7 =  edm::hrRealTime() -s7;


  Matrix mrh;
  s8 = edm::hrRealTime();
  similarity(mrh,j1,im1);
  s8 =  edm::hrRealTime() -s8;


 
  SymMatrix lh = ROOT::Math::SimilarityT(j2, is1);

  Matrix mlh;  similarityT(mlh,j2,im1);


 


  SymMatrix a; 
  s1 = edm::hrRealTime();
  mult(a,lh,rh);
  s1 =  edm::hrRealTime() -s1;

  SymMatrix b;
  s2 = edm::hrRealTime();
  ROOT::Math::AssignSym::Evaluate(b, lh*rh);
  s2 =  edm::hrRealTime() -s2;

  Matrix m0 = b;


  Matrix m;
  s3 = edm::hrRealTime();
  m = mlh*mrh;
  s3 =  edm::hrRealTime() -s3;

  SymMatrix sm;
  s4 = edm::hrRealTime();
  ROOT::Math::AssignSym::Evaluate(sm,mlh*mrh);
  s4 =  edm::hrRealTime() -s4;

  Matrix m2;
  s5 = edm::hrRealTime();
  mult(m2,mlh,mrh);
  s5 =  edm::hrRealTime() -s5;


  SymMatrix smm; ROOT::Math::AssignSym::Evaluate(smm,m);
  SymMatrix smm2; ROOT::Math::AssignSym::Evaluate(smm2,m2);

  if (print) {
    if (!isSym(im1))  std::cout << " im is not sym" << std::endl;
    if (!isSym(mrh)) std::cout << " rh is not sym" << std::endl;
    if (!isSym(mlh)) std::cout << " lh is not sym" << std::endl;
    if (!isSym(m)) std::cout << " m is not sym" << std::endl;
    if (!isSym(m2)) std::cout << "m2 is not sym" << std::endl;

    std::cout << "eps sim  " << eps(rh,mrh) << std::endl;
    std::cout << "eps sim  " << eps(rh,rh2) << std::endl;
    std::cout << "eps simT " << eps(lh,mlh) << std::endl;
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
    std::cout << "std  sim   " << s7 << std::endl;
    std::cout << "loop sim   " << s8 << std::endl;
  
  }

}
 
int main() {
  edm::HRTimeType s1=0, s2=0, s3=0, s4=0, s5=0, s6=0, s7=0, s8=0;
  go<double,3>(s1,s2,s3,s4,s5,s6,s7,s8, true);
 

  go<double,15>(s1,s2,s3,s4,s5,s6,s7,s8, true);

  edm::HRTimeType t1=0; edm::HRTimeType t2=0;  edm::HRTimeType t3=0; edm::HRTimeType t4=0; edm::HRTimeType t5=0;
  edm::HRTimeType t6=0; edm::HRTimeType t7=0;  edm::HRTimeType t8=0;

  for (int  i=0; i!=50000; ++i) {
    go<double,15>(s1,s2,s3,s4,s5,s6,s7,s8, false);
    t1+=s1; t2+=s2; t3+=s3; t4+=s4; t5+=s5;  t6+=s6; t7+=s7; t8+=s8;
  }
  std::cout << "sym mult   " << t1/50000 << std::endl;
  std::cout << "sym    *   " << t2/50000 << std::endl;
  std::cout << "mat    *   " << t3/50000 << std::endl;
  std::cout << "mat as sym " << t4/50000 << std::endl;
  std::cout << "mat mult   " << t5/50000 << std::endl;
  std::cout << "sym  sim " << t6/50000 << std::endl;
  std::cout << "std  sim " << t7/50000 << std::endl;
  std::cout << "loop sim " << t8/50000 << std::endl;
 
  return 0;
}

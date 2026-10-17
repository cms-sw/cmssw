#define SMATRIX_USE_CONSTEXPR
#include "Math/SMatrix.h"
#include <cmath>
#include <cstddef>
#include <random>

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
    struct AssignAsSym {
      /// assign a symmetric matrix from an expression
      template <class T, IndexType D, class A, class R>
      static void Evaluate(SMatrix<T, D, D, MatRepStd<T, D> >& lhs, const Expr<A, T, D, D, R>& rhs) {
        // for(unsigned int i=0; i<D*D; ++i) lhs.fRep[i] = rhs.apply(i);

        // in principle this will do only half+D evaluations (and is ~4 times faster than the above for a multiplication)
        for (IndexType i = 0; i < D; ++i)
          // storage of symmetric matrix is in lower block
          for (IndexType j = 0; j <= i; ++j)
            lhs(i, j) = rhs(i, j);
        // symmetrize
        for (IndexType i = 0; i != D - 1; ++i)
          for (IndexType j = i + 1; j != D; ++j)
            lhs(i, j) = lhs(j, i);
      }
      /// assign the "symmetric" matric  from a general matrix
      template <class T, IndexType D, class R>
      static void Evaluate(SMatrix<T, D, D, MatRepStd<T, D> >& lhs, const SMatrix<T, D, D, R>& rhs) {
        for (IndexType i = 0; i != D * D; ++i)
          lhs.fRep[i] = rhs.apply(i);
        /*
       // useful only if we do not store the upper triangle
       for( IndexType i=0; i<D; ++i)
       // storage of symmetric matrix is in lower block
       for( IndexType j=0; j<=i; ++j) 
       lhs(i,j) = rhs(i,j);
       for (IndexType i=0; i!=D-1; ++i) 
	 for (IndexType j=i+1; j!=D; ++j)
	   lhs(i,j) = lhs(j,i); 
       */
      }
    };  // struct AssignAsSym

    template <class T, IndexType D1, IndexType D2, class R>
    inline SMatrix<T, D1, D1, MatRepSym<T, D1> > Similarity1(const SMatrix<T, D1, D2, R>& lhs,
                                                             const SMatrix<T, D2, D2, MatRepStd<T, D2> >& rhs) {
      SMatrix<T, D1, D2, MatRepStd<T, D1, D2> > tmp = lhs * rhs;
      typedef SMatrix<T, D1, D1, MatRepSym<T, D1> > SMatrixSym;
      SMatrixSym mret{SMatrixNoInit{}};
      AssignSym::Evaluate(mret, tmp * Transpose(lhs));
      return mret;
    }

    template <class T, IndexType D1, IndexType D2>
    inline SMatrix<T, D1, D1, MatRepStd<T, D1> > Similarity2(const SMatrix<T, D1, D2, MatRepStd<T, D1, D2> >& lhs,
                                                             const SMatrix<T, D2, D2, MatRepStd<T, D2> >& rhs) {
      SMatrix<T, D1, D2, MatRepStd<T, D1, D2> > tmp = lhs * rhs;
      typedef SMatrix<T, D1, D1, MatRepStd<T, D1> > SMatrixSym;
      SMatrixSym mret = SMatrixNoInit();
      AssignAsSym::Evaluate(mret, tmp * Transpose(lhs));
      return mret;
    }
  }  // namespace Math
}  // namespace ROOT

// U(i,k) * A(k,l) * U(j,l)
template <typename T, IndexType D1, IndexType D2>
inline void similarity(ROOT::Math::SMatrix<T, D1, D1, ROOT::Math::MatRepStd<T, D1> >& b,
                       ROOT::Math::SMatrix<T, D1, D2, ROOT::Math::MatRepStd<T, D1, D2> > const& u,
                       ROOT::Math::SMatrix<T, D2, D2, ROOT::Math::MatRepStd<T, D2> > const& a) {
  // brute force loop
  for (IndexType i = 0; i != D1; ++i)
    for (IndexType j = 0; j <= i; ++j)
      for (IndexType k = 0; k != D2; ++k)
        for (IndexType l = 0; l != D2; ++l)
          b(i, j) += u(i, k) * a(k, l) * u(j, l);

  /*
  for (IndexType is=0; is<D1; is+=4) {
    IndexType ie=std::min(is+4,D1);
    //for (IndexType js=0; js<=is; js+=4) {
      for (IndexType ks=0; ks<D2; ks+=4) { 
	IndexType ke=std::min(ks+4,D2);
	for (IndexType i=is; i!=ie; ++i) 
	  for (IndexType j=0; j<=i; ++j) 
	    //	  for (IndexType j=js; j<=std::min(js+3,i); ++j) 
	    for (IndexType k=ks; k!=ke; ++k) 
	      for (IndexType l=0; l!=D2; ++l) 
		b(i,j) += u(i,k)*a(k,l)*u(j,l);
      }
      //}
  }
  */

  //	for (IndexType l=0; l<=k; ++l)
  //  b(i,j) += (u(i,k)*u(j,l)+u(i,l)*u(j,k))*a(k,l);
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

  for (IndexType i = 0; i != D1 - 1; ++i)
    for (IndexType j = i + 1; j != D1; ++j)
      b(i, j) = b(j, i);
}

template <typename M1, typename M2>
double eps(M1 const& m1, M2 const& m2) {
  IndexType N = M1::kRows;
  double ret = 0.;
  for (IndexType i = 0; i != N; ++i)
    for (IndexType j = 0; j != N; ++j)
      ret = std::max(ret, std::abs(m1(i, j) - m2(i, j)));
  return ret;
}

template <typename M>
bool isSym(M const& m) {
  IndexType N = M::kRows;
  for (IndexType i = 0; i != N; ++i)
    for (IndexType j = 0; j <= i; ++j)
      if (m(i, j) != m(j, i))
        return false;
  return true;
}

#include <iostream>
#include <benchmark/benchmark.h>

namespace {
  std::mt19937 eng;
  std::uniform_real_distribution<double> rgen(-5., 5.);

  inline double rr() { return rgen(eng); }

  template <typename T, IndexType D1, IndexType D2>
  inline void fillRandom(ROOT::Math::SMatrix<T, D1, D2, ROOT::Math::MatRepStd<T, D1, D2> >& a) {
    for (IndexType i = 0; i != D1; ++i) {
      for (IndexType j = 0; j != D2; ++j)
        a(i, j) = rr();
    }
  }
  template <typename T, IndexType N>
  inline void fillRandomSym(ROOT::Math::SMatrix<T, N, N, ROOT::Math::MatRepStd<T, N> >& a) {
    for (IndexType i = 0; i != N; ++i) {
      for (IndexType j = 0; j <= i; ++j)
        a(i, j) = rr();
    }
    for (IndexType i = 0; i != N - 1; ++i)
      for (IndexType j = i + 1; j != N; ++j)
        a(i, j) = a(j, i);
  }
}  // namespace

bool ok = true;

template <typename T, IndexType D1, IndexType D2>
void go(bool print) {
  typedef ROOT::Math::SMatrix<T, D1, D2, ROOT::Math::MatRepStd<T, D1, D2> > JMatrix;
  typedef ROOT::Math::SMatrix<T, D1, D1, ROOT::Math::MatRepStd<T, D1, D1> > Matrix1;
  typedef ROOT::Math::SMatrix<T, D1, D1, ROOT::Math::MatRepSym<T, D1> > SymMatrix1;
  typedef ROOT::Math::SMatrix<T, D2, D2, ROOT::Math::MatRepStd<T, D2, D2> > Matrix2;
  typedef ROOT::Math::SMatrix<T, D2, D2, ROOT::Math::MatRepSym<T, D2> > SymMatrix2;

  JMatrix lh;
  fillRandom(lh);
  Matrix2 rh;
  fillRandomSym(rh);

  SymMatrix2 srh;
  ROOT::Math::AssignSym::Evaluate(srh, rh);

  SymMatrix1 res1;
  res1 = ROOT::Math::Similarity(lh, srh);

  SymMatrix1 res2;
  res2 = ROOT::Math::Similarity1(lh, rh);

  Matrix1 res3;
  res3 = ROOT::Math::Similarity2(lh, rh);

  Matrix1 res4;
  similarity(res4, lh, rh);

  if (print) {
    if (!isSym(rh))
      std::cout << " rh is not sym" << std::endl;
    if (!isSym(res1))
      std::cout << " res1 is not sym" << std::endl;
    if (!isSym(res2))
      std::cout << " res2 is not sym" << std::endl;
    if (!isSym(res3))
      std::cout << " res3 is not sym" << std::endl;
    if (!isSym(res4))
      std::cout << " res4 is not sym" << std::endl;

    std::cout << D1 << "x" << D2 << std::endl;
    std::cout << "eps sim  " << eps(res1, res2) << std::endl;
    std::cout << "eps std  " << eps(res1, res3) << std::endl;
    std::cout << "eps loop  " << eps(res1, res4) << std::endl;
  }

  ok &= isSym(rh) && isSym(res1) && isSym(res2) && isSym(res3) && isSym(res4);
}

namespace {
  template <typename T, IndexType D1, IndexType D2>
  struct Data {
    typedef ROOT::Math::SMatrix<T, D1, D2, ROOT::Math::MatRepStd<T, D1, D2> > JMatrix;
    typedef ROOT::Math::SMatrix<T, D1, D1, ROOT::Math::MatRepStd<T, D1, D1> > Matrix1;
    typedef ROOT::Math::SMatrix<T, D1, D1, ROOT::Math::MatRepSym<T, D1> > SymMatrix1;
    typedef ROOT::Math::SMatrix<T, D2, D2, ROOT::Math::MatRepStd<T, D2, D2> > Matrix2;
    typedef ROOT::Math::SMatrix<T, D2, D2, ROOT::Math::MatRepSym<T, D2> > SymMatrix2;

    Data() {
      fillRandom(lh);
      fillRandomSym(rh);
      ROOT::Math::AssignSym::Evaluate(srh, rh);
    }

    JMatrix lh;
    Matrix2 rh;
    SymMatrix2 srh;
  };

  template <typename T, IndexType D1, IndexType D2>
  void BM_rootSimilarity(benchmark::State& state) {
    Data<T, D1, D2> d;
    typename Data<T, D1, D2>::SymMatrix1 res;
    for (auto _ : state) {
      res = ROOT::Math::Similarity(d.lh, d.srh);
      benchmark::DoNotOptimize(res);
    }
  }

  template <typename T, IndexType D1, IndexType D2>
  void BM_symSimilarity(benchmark::State& state) {
    Data<T, D1, D2> d;
    typename Data<T, D1, D2>::SymMatrix1 res;
    for (auto _ : state) {
      res = ROOT::Math::Similarity1(d.lh, d.rh);
      benchmark::DoNotOptimize(res);
    }
  }

  template <typename T, IndexType D1, IndexType D2>
  void BM_stdSimilarity(benchmark::State& state) {
    Data<T, D1, D2> d;
    typename Data<T, D1, D2>::Matrix1 res;
    for (auto _ : state) {
      res = ROOT::Math::Similarity2(d.lh, d.rh);
      benchmark::DoNotOptimize(res);
    }
  }

  template <typename T, IndexType D1, IndexType D2>
  void BM_loopSimilarity(benchmark::State& state) {
    Data<T, D1, D2> d;
    typename Data<T, D1, D2>::Matrix1 res;
    for (auto _ : state) {
      similarity(res, d.lh, d.rh);
      benchmark::DoNotOptimize(res);
    }
  }

#define SIMILARITY_BENCHMARKS(T, D1, D2)            \
  BENCHMARK_TEMPLATE(BM_rootSimilarity, T, D1, D2); \
  BENCHMARK_TEMPLATE(BM_symSimilarity, T, D1, D2);  \
  BENCHMARK_TEMPLATE(BM_stdSimilarity, T, D1, D2);  \
  BENCHMARK_TEMPLATE(BM_loopSimilarity, T, D1, D2)

  SIMILARITY_BENCHMARKS(double, 3, 3);
  SIMILARITY_BENCHMARKS(double, 5, 5);
  SIMILARITY_BENCHMARKS(double, 5, 15);
  SIMILARITY_BENCHMARKS(double, 15, 15);
}  // namespace

int main(int argc, char* argv[]) {
  ok = true;
  go<double, 3, 3>(true);
  go<double, 5, 5>(true);
  go<double, 5, 15>(true);
  go<double, 15, 15>(true);
  std::cout << std::endl;

  if (ok)
    std::cout << " OK " << std::endl;

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();

  return 0;
}

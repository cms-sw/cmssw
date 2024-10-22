#ifndef DataFormat_Math_choleskyInversion_h
#define DataFormat_Math_choleskyInversion_h

#include <cmath>

#include <Eigen/Core>

namespace math {
  namespace cholesky {

    template <typename M1, typename M2, int N = M2::ColsAtCompileTime>
// without this: either does not compile or compiles and then fails silently at runtime
#ifdef __CUDACC__
    __host__ __device__
#endif
        inline constexpr void
        invertNN(M1 const& src, M2& dst) {

      // origin: CERNLIB

      using T = typename M2::Scalar;

      T a[N][N];
      for (int i = 0; i < N; ++i) {
        a[i][i] = src(i, i);
        for (int j = i + 1; j < N; ++j)
          a[j][i] = src(i, j);
      }

      for (int j = 0; j < N; ++j) {
        a[j][j] = T(1.) / a[j][j];
        int jp1 = j + 1;
        for (int l = jp1; l < N; ++l) {
          a[j][l] = a[j][j] * a[l][j];
          T s1 = -a[l][jp1];
          for (int i = 0; i < jp1; ++i)
            s1 += a[l][i] * a[i][jp1];
          a[l][jp1] = -s1;
        }
      }

      if constexpr (N == 1) {
        dst(0, 0) = a[0][0];
        return;
      }
      a[0][1] = -a[0][1];
      a[1][0] = a[0][1] * a[1][1];
      for (int j = 2; j < N; ++j) {
        int jm1 = j - 1;
        for (int k = 0; k < jm1; ++k) {
          T s31 = a[k][j];
          for (int i = k; i < jm1; ++i)
            s31 += a[k][i + 1] * a[i + 1][j];
          a[k][j] = -s31;
          a[j][k] = -s31 * a[j][j];
        }
        a[jm1][j] = -a[jm1][j];
        a[j][jm1] = a[jm1][j] * a[j][j];
      }

      int j = 0;
      while (j < N - 1) {
        T s33 = a[j][j];
        for (int i = j + 1; i < N; ++i)
          s33 += a[j][i] * a[i][j];
        dst(j, j) = s33;

        ++j;
        for (int k = 0; k < j; ++k) {
          T s32 = 0;
          for (int i = j; i < N; ++i)
            s32 += a[k][i] * a[i][j];
          dst(k, j) = dst(j, k) = s32;
        }
      }
      dst(j, j) = a[j][j];
    }

    /**
 * fully inlined specialized code to perform the inversion of a
 * positive defined matrix of rank up to 6.
 *
 * adapted from ROOT::Math::CholeskyDecomp
 * originally by
 * @author Manuel Schiller
 * @date Aug 29 2008
 *
 *
 */

    template <typename M1, typename M2>
    inline constexpr void __attribute__((always_inline)) invert11(M1 const& src, M2& dst) {
      using F = decltype(src(0, 0));
      dst(0, 0) = F(1.0) / src(0, 0);
    }

    template <typename M1, typename M2>
    inline constexpr void __attribute__((always_inline)) invert22(M1 const& src, M2& dst) {
      using F = decltype(src(0, 0));
      auto luc0 = F(1.0) / src(0, 0);
      auto luc1 = src(1, 0) * src(1, 0) * luc0;
      auto luc2 = F(1.0) / (src(1, 1) - luc1);

      auto li21 = luc1 * luc0 * luc2;

      dst(0, 0) = li21 + luc0;
      dst(1, 0) = -src(1, 0) * luc0 * luc2;
      dst(1, 1) = luc2;
    }

    template <typename M1, typename M2>
    inline constexpr void __attribute__((always_inline)) invert33(M1 const& src, M2& dst) {
      using F = decltype(src(0, 0));
      auto luc0 = F(1.0) / src(0, 0);
      auto luc1 = src(1, 0);
      auto luc2 = src(1, 1) - luc0 * luc1 * luc1;
      luc2 = F(1.0) / luc2;
      auto luc3 = src(2, 0);
      auto luc4 = (src(2, 1) - luc0 * luc1 * luc3);
      auto luc5 = src(2, 2) - (luc0 * luc3 * luc3 + (luc2 * luc4) * luc4);
      luc5 = F(1.0) / luc5;

      auto li21 = -luc0 * luc1;
      auto li32 = -(luc2 * luc4);
      auto li31 = (luc1 * (luc2 * luc4) - luc3) * luc0;

      dst(0, 0) = luc5 * li31 * li31 + li21 * li21 * luc2 + luc0;
      dst(1, 0) = luc5 * li31 * li32 + li21 * luc2;
      dst(1, 1) = luc5 * li32 * li32 + luc2;
      dst(2, 0) = luc5 * li31;
      dst(2, 1) = luc5 * li32;
      dst(2, 2) = luc5;
    }

    template <typename M1, typename M2>
    inline constexpr void __attribute__((always_inline)) invert44(M1 const& src, M2& dst) {
      using F = decltype(src(0, 0));
      auto luc0 = F(1.0) / src(0, 0);
      auto luc1 = src(1, 0);
      auto luc2 = src(1, 1) - luc0 * luc1 * luc1;
      luc2 = F(1.0) / luc2;
      auto luc3 = src(2, 0);
      auto luc4 = (src(2, 1) - luc0 * luc1 * luc3);
      auto luc5 = src(2, 2) - (luc0 * luc3 * luc3 + luc2 * luc4 * luc4);
      luc5 = F(1.0) / luc5;
      auto luc6 = src(3, 0);
      auto luc7 = (src(3, 1) - luc0 * luc1 * luc6);
      auto luc8 = (src(3, 2) - luc0 * luc3 * luc6 - luc2 * luc4 * luc7);
      auto luc9 = src(3, 3) - (luc0 * luc6 * luc6 + luc2 * luc7 * luc7 + luc8 * (luc8 * luc5));
      luc9 = F(1.0) / luc9;

      auto li21 = -luc1 * luc0;
      auto li32 = -luc2 * luc4;
      auto li31 = (luc1 * (luc2 * luc4) - luc3) * luc0;
      auto li43 = -(luc8 * luc5);
      auto li42 = (luc4 * luc8 * luc5 - luc7) * luc2;
      auto li41 = (-luc1 * (luc2 * luc4) * (luc8 * luc5) + luc1 * (luc2 * luc7) + luc3 * (luc8 * luc5) - luc6) * luc0;

      dst(0, 0) = luc9 * li41 * li41 + luc5 * li31 * li31 + luc2 * li21 * li21 + luc0;
      dst(1, 0) = luc9 * li41 * li42 + luc5 * li31 * li32 + luc2 * li21;
      dst(1, 1) = luc9 * li42 * li42 + luc5 * li32 * li32 + luc2;
      dst(2, 0) = luc9 * li41 * li43 + luc5 * li31;
      dst(2, 1) = luc9 * li42 * li43 + luc5 * li32;
      dst(2, 2) = luc9 * li43 * li43 + luc5;
      dst(3, 0) = luc9 * li41;
      dst(3, 1) = luc9 * li42;
      dst(3, 2) = luc9 * li43;
      dst(3, 3) = luc9;
    }

    template <typename M1, typename M2>
    inline constexpr void __attribute__((always_inline)) invert55(M1 const& src, M2& dst) {
      using F = decltype(src(0, 0));
      auto luc0 = F(1.0) / src(0, 0);
      auto luc1 = src(1, 0);
      auto luc2 = src(1, 1) - luc0 * luc1 * luc1;
      luc2 = F(1.0) / luc2;
      auto luc3 = src(2, 0);
      auto luc4 = (src(2, 1) - luc0 * luc1 * luc3);
      auto luc5 = src(2, 2) - (luc0 * luc3 * luc3 + luc2 * luc4 * luc4);
      luc5 = F(1.0) / luc5;
      auto luc6 = src(3, 0);
      auto luc7 = (src(3, 1) - luc0 * luc1 * luc6);
      auto luc8 = (src(3, 2) - luc0 * luc3 * luc6 - luc2 * luc4 * luc7);
      auto luc9 = src(3, 3) - (luc0 * luc6 * luc6 + luc2 * luc7 * luc7 + luc8 * (luc8 * luc5));
      luc9 = F(1.0) / luc9;
      auto luc10 = src(4, 0);
      auto luc11 = (src(4, 1) - luc0 * luc1 * luc10);
      auto luc12 = (src(4, 2) - luc0 * luc3 * luc10 - luc2 * luc4 * luc11);
      auto luc13 = (src(4, 3) - luc0 * luc6 * luc10 - luc2 * luc7 * luc11 - luc5 * luc8 * luc12);
      auto luc14 =
          src(4, 4) - (luc0 * luc10 * luc10 + luc2 * luc11 * luc11 + luc5 * luc12 * luc12 + luc9 * luc13 * luc13);
      luc14 = F(1.0) / luc14;

      auto li21 = -luc1 * luc0;
      auto li32 = -luc2 * luc4;
      auto li31 = (luc1 * (luc2 * luc4) - luc3) * luc0;
      auto li43 = -(luc8 * luc5);
      auto li42 = (luc4 * luc8 * luc5 - luc7) * luc2;
      auto li41 = (-luc1 * (luc2 * luc4) * (luc8 * luc5) + luc1 * (luc2 * luc7) + luc3 * (luc8 * luc5) - luc6) * luc0;
      auto li54 = -luc13 * luc9;
      auto li53 = (luc13 * luc8 * luc9 - luc12) * luc5;
      auto li52 = (-luc4 * luc8 * luc13 * luc5 * luc9 + luc4 * luc12 * luc5 + luc7 * luc13 * luc9 - luc11) * luc2;
      auto li51 = (luc1 * luc4 * luc8 * luc13 * luc2 * luc5 * luc9 - luc13 * luc8 * luc3 * luc9 * luc5 -
                   luc12 * luc4 * luc1 * luc2 * luc5 - luc13 * luc7 * luc1 * luc9 * luc2 + luc11 * luc1 * luc2 +
                   luc12 * luc3 * luc5 + luc13 * luc6 * luc9 - luc10) *
                  luc0;

      dst(0, 0) = luc14 * li51 * li51 + luc9 * li41 * li41 + luc5 * li31 * li31 + luc2 * li21 * li21 + luc0;
      dst(1, 0) = luc14 * li51 * li52 + luc9 * li41 * li42 + luc5 * li31 * li32 + luc2 * li21;
      dst(1, 1) = luc14 * li52 * li52 + luc9 * li42 * li42 + luc5 * li32 * li32 + luc2;
      dst(2, 0) = luc14 * li51 * li53 + luc9 * li41 * li43 + luc5 * li31;
      dst(2, 1) = luc14 * li52 * li53 + luc9 * li42 * li43 + luc5 * li32;
      dst(2, 2) = luc14 * li53 * li53 + luc9 * li43 * li43 + luc5;
      dst(3, 0) = luc14 * li51 * li54 + luc9 * li41;
      dst(3, 1) = luc14 * li52 * li54 + luc9 * li42;
      dst(3, 2) = luc14 * li53 * li54 + luc9 * li43;
      dst(3, 3) = luc14 * li54 * li54 + luc9;
      dst(4, 0) = luc14 * li51;
      dst(4, 1) = luc14 * li52;
      dst(4, 2) = luc14 * li53;
      dst(4, 3) = luc14 * li54;
      dst(4, 4) = luc14;
    }

    template <typename M1, typename M2>
    inline constexpr void __attribute__((always_inline)) invert66(M1 const& src, M2& dst) {
      using F = decltype(src(0, 0));
      auto luc0 = F(1.0) / src(0, 0);
      auto luc1 = src(1, 0);
      auto luc2 = src(1, 1) - luc0 * luc1 * luc1;
      luc2 = F(1.0) / luc2;
      auto luc3 = src(2, 0);
      auto luc4 = (src(2, 1) - luc0 * luc1 * luc3);
      auto luc5 = src(2, 2) - (luc0 * luc3 * luc3 + luc2 * luc4 * luc4);
      luc5 = F(1.0) / luc5;
      auto luc6 = src(3, 0);
      auto luc7 = (src(3, 1) - luc0 * luc1 * luc6);
      auto luc8 = (src(3, 2) - luc0 * luc3 * luc6 - luc2 * luc4 * luc7);
      auto luc9 = src(3, 3) - (luc0 * luc6 * luc6 + luc2 * luc7 * luc7 + luc8 * (luc8 * luc5));
      luc9 = F(1.0) / luc9;
      auto luc10 = src(4, 0);
      auto luc11 = (src(4, 1) - luc0 * luc1 * luc10);
      auto luc12 = (src(4, 2) - luc0 * luc3 * luc10 - luc2 * luc4 * luc11);
      auto luc13 = (src(4, 3) - luc0 * luc6 * luc10 - luc2 * luc7 * luc11 - luc5 * luc8 * luc12);
      auto luc14 =
          src(4, 4) - (luc0 * luc10 * luc10 + luc2 * luc11 * luc11 + luc5 * luc12 * luc12 + luc9 * luc13 * luc13);
      luc14 = F(1.0) / luc14;
      auto luc15 = src(5, 0);
      auto luc16 = (src(5, 1) - luc0 * luc1 * luc15);
      auto luc17 = (src(5, 2) - luc0 * luc3 * luc15 - luc2 * luc4 * luc16);
      auto luc18 = (src(5, 3) - luc0 * luc6 * luc15 - luc2 * luc7 * luc16 - luc5 * luc8 * luc17);
      auto luc19 =
          (src(5, 4) - luc0 * luc10 * luc15 - luc2 * luc11 * luc16 - luc5 * luc12 * luc17 - luc9 * luc13 * luc18);
      auto luc20 = src(5, 5) - (luc0 * luc15 * luc15 + luc2 * luc16 * luc16 + luc5 * luc17 * luc17 +
                                luc9 * luc18 * luc18 + luc14 * luc19 * luc19);
      luc20 = F(1.0) / luc20;

      auto li21 = -luc1 * luc0;
      auto li32 = -luc2 * luc4;
      auto li31 = (luc1 * (luc2 * luc4) - luc3) * luc0;
      auto li43 = -(luc8 * luc5);
      auto li42 = (luc4 * luc8 * luc5 - luc7) * luc2;
      auto li41 = (-luc1 * (luc2 * luc4) * (luc8 * luc5) + luc1 * (luc2 * luc7) + luc3 * (luc8 * luc5) - luc6) * luc0;
      auto li54 = -luc13 * luc9;
      auto li53 = (luc13 * luc8 * luc9 - luc12) * luc5;
      auto li52 = (-luc4 * luc8 * luc13 * luc5 * luc9 + luc4 * luc12 * luc5 + luc7 * luc13 * luc9 - luc11) * luc2;
      auto li51 = (luc1 * luc4 * luc8 * luc13 * luc2 * luc5 * luc9 - luc13 * luc8 * luc3 * luc9 * luc5 -
                   luc12 * luc4 * luc1 * luc2 * luc5 - luc13 * luc7 * luc1 * luc9 * luc2 + luc11 * luc1 * luc2 +
                   luc12 * luc3 * luc5 + luc13 * luc6 * luc9 - luc10) *
                  luc0;

      auto li65 = -luc19 * luc14;
      auto li64 = (luc19 * luc14 * luc13 - luc18) * luc9;
      auto li63 =
          (-luc8 * luc13 * (luc19 * luc14) * luc9 + luc8 * luc9 * luc18 + luc12 * (luc19 * luc14) - luc17) * luc5;
      auto li62 = (luc4 * (luc8 * luc9) * luc13 * luc5 * (luc19 * luc14) - luc18 * luc4 * (luc8 * luc9) * luc5 -
                   luc19 * luc12 * luc4 * luc14 * luc5 - luc19 * luc13 * luc7 * luc14 * luc9 + luc17 * luc4 * luc5 +
                   luc18 * luc7 * luc9 + luc19 * luc11 * luc14 - luc16) *
                  luc2;
      auto li61 =
          (-luc19 * luc13 * luc8 * luc4 * luc1 * luc2 * luc5 * luc9 * luc14 +
           luc18 * luc8 * luc4 * luc1 * luc2 * luc5 * luc9 + luc19 * luc12 * luc4 * luc1 * luc2 * luc5 * luc14 +
           luc19 * luc13 * luc7 * luc1 * luc2 * luc9 * luc14 + luc19 * luc13 * luc8 * luc3 * luc5 * luc9 * luc14 -
           luc17 * luc4 * luc1 * luc2 * luc5 - luc18 * luc7 * luc1 * luc2 * luc9 - luc19 * luc11 * luc1 * luc2 * luc14 -
           luc18 * luc8 * luc3 * luc5 * luc9 - luc19 * luc12 * luc3 * luc5 * luc14 -
           luc19 * luc13 * luc6 * luc9 * luc14 + luc16 * luc1 * luc2 + luc17 * luc3 * luc5 + luc18 * luc6 * luc9 +
           luc19 * luc10 * luc14 - luc15) *
          luc0;

      dst(0, 0) = luc20 * li61 * li61 + luc14 * li51 * li51 + luc9 * li41 * li41 + luc5 * li31 * li31 +
                  luc2 * li21 * li21 + luc0;
      dst(1, 0) = luc20 * li61 * li62 + luc14 * li51 * li52 + luc9 * li41 * li42 + luc5 * li31 * li32 + luc2 * li21;
      dst(1, 1) = luc20 * li62 * li62 + luc14 * li52 * li52 + luc9 * li42 * li42 + luc5 * li32 * li32 + luc2;
      dst(2, 0) = luc20 * li61 * li63 + luc14 * li51 * li53 + luc9 * li41 * li43 + luc5 * li31;
      dst(2, 1) = luc20 * li62 * li63 + luc14 * li52 * li53 + luc9 * li42 * li43 + luc5 * li32;
      dst(2, 2) = luc20 * li63 * li63 + luc14 * li53 * li53 + luc9 * li43 * li43 + luc5;
      dst(3, 0) = luc20 * li61 * li64 + luc14 * li51 * li54 + luc9 * li41;
      dst(3, 1) = luc20 * li62 * li64 + luc14 * li52 * li54 + luc9 * li42;
      dst(3, 2) = luc20 * li63 * li64 + luc14 * li53 * li54 + luc9 * li43;
      dst(3, 3) = luc20 * li64 * li64 + luc14 * li54 * li54 + luc9;
      dst(4, 0) = luc20 * li61 * li65 + luc14 * li51;
      dst(4, 1) = luc20 * li62 * li65 + luc14 * li52;
      dst(4, 2) = luc20 * li63 * li65 + luc14 * li53;
      dst(4, 3) = luc20 * li64 * li65 + luc14 * li54;
      dst(4, 4) = luc20 * li65 * li65 + luc14;
      dst(5, 0) = luc20 * li61;
      dst(5, 1) = luc20 * li62;
      dst(5, 2) = luc20 * li63;
      dst(5, 3) = luc20 * li64;
      dst(5, 4) = luc20 * li65;
      dst(5, 5) = luc20;
    }

    template <typename M>
    inline constexpr void symmetrize11(M& dst) {}

    template <typename M>
    inline constexpr void symmetrize22(M& dst) {
      dst(0, 1) = dst(1, 0);
    }

    template <typename M>
    inline constexpr void symmetrize33(M& dst) {
      symmetrize22(dst);
      dst(0, 2) = dst(2, 0);
      dst(1, 2) = dst(2, 1);
    }

    template <typename M>
    inline constexpr void symmetrize44(M& dst) {
      symmetrize33(dst);
      dst(0, 3) = dst(3, 0);
      dst(1, 3) = dst(3, 1);
      dst(2, 3) = dst(3, 2);
    }

    template <typename M>
    inline constexpr void symmetrize55(M& dst) {
      symmetrize44(dst);
      dst(0, 4) = dst(4, 0);
      dst(1, 4) = dst(4, 1);
      dst(2, 4) = dst(4, 2);
      dst(3, 4) = dst(4, 3);
    }

    template <typename M>
    inline constexpr void symmetrize66(M& dst) {
      symmetrize55(dst);
      dst(0, 5) = dst(5, 0);
      dst(1, 5) = dst(5, 1);
      dst(2, 5) = dst(5, 2);
      dst(3, 5) = dst(5, 3);
      dst(4, 5) = dst(5, 4);
    }

    template <typename M1, typename M2, int N>
    struct Inverter {
      static constexpr void eval(M1 const& src, M2& dst) { dst = src.inverse(); }
    };

    template <typename M1, typename M2>
    struct Inverter<M1, M2, 1> {
      static constexpr void eval(M1 const& src, M2& dst) { invert11(src, dst); }
    };

    template <typename M1, typename M2>
    struct Inverter<M1, M2, 2> {
      static constexpr void __attribute__((always_inline)) eval(M1 const& src, M2& dst) {
        invert22(src, dst);
        symmetrize22(dst);
      }
    };

    template <typename M1, typename M2>
    struct Inverter<M1, M2, 3> {
      static constexpr void __attribute__((always_inline)) eval(M1 const& src, M2& dst) {
        invert33(src, dst);
        symmetrize33(dst);
      }
    };

    template <typename M1, typename M2>
    struct Inverter<M1, M2, 4> {
      static constexpr void __attribute__((always_inline)) eval(M1 const& src, M2& dst) {
        invert44(src, dst);
        symmetrize44(dst);
      }
    };

    template <typename M1, typename M2>
    struct Inverter<M1, M2, 5> {
      static constexpr void __attribute__((always_inline)) eval(M1 const& src, M2& dst) {
        invert55(src, dst);
        symmetrize55(dst);
      }
    };

    template <typename M1, typename M2>
    struct Inverter<M1, M2, 6> {
      static constexpr void __attribute__((always_inline)) eval(M1 const& src, M2& dst) {
        invert66(src, dst);
        symmetrize66(dst);
      }
    };

    // Eigen interface
    template <typename M1, typename M2>
    inline constexpr void __attribute__((always_inline)) invert(M1 const& src, M2& dst) {
      if constexpr (M2::ColsAtCompileTime < 7)
        Inverter<M1, M2, M2::ColsAtCompileTime>::eval(src, dst);
      else
        invertNN(src, dst);
    }

  }  // namespace cholesky
}  // namespace math

#endif  // DataFormat_Math_choleskyInversion_h

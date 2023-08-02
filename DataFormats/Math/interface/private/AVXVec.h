#ifndef DataFormat_Math_AVXVec_H
#define DataFormat_Math_AVXVec_H

// in principle it should not be used alone
// only as part of SSEVec
namespace mathSSE {

  template <>
  union Vec4<double> {
    typedef __m256d nativeType;
    __m256d vec;
    double __attribute__((aligned(32))) arr[4];
    OldVec<double> o;

    Vec4(__m256d ivec) : vec(ivec) {}

    Vec4(OldVec<double> const& ivec) : o(ivec) {}

    Vec4() { vec = _mm256_setzero_pd(); }

    inline Vec4(Vec4<float> ivec) { vec = _mm256_cvtps_pd(ivec.vec); }

    explicit Vec4(double f1) { set1(f1); }

    Vec4(double f1, double f2, double f3, double f4 = 0) {
      arr[0] = f1;
      arr[1] = f2;
      arr[2] = f3;
      arr[3] = f4;
    }

    Vec4(Vec2<double> ivec0, Vec2<double> ivec1) {
      vec = _mm256_set_m128d(ivec1.vec, ivec0.vec);
    }

    Vec4(Vec2<double> ivec0, double f3, double f4 = 0) {
      vec = _mm256_insertf128_pd(vec, ivec0.vec, 0);
      arr[2] = f3;
      arr[3] = f4;
    }

    Vec4(Vec2<double> ivec0) {
      vec = _mm256_setzero_pd();
      vec = _mm256_insertf128_pd(vec, ivec0.vec, 0);
    }

    // for masking
    void setMask(unsigned int m1, unsigned int m2, unsigned int m3, unsigned int m4) {
      Mask4<double> mask(m1, m2, m3, m4);
      vec = mask.vec;
    }

    void set(double f1, double f2, double f3, double f4 = 0) { vec = _mm256_set_pd(f4, f3, f2, f1); }

    void set1(double f1) { vec = _mm256_set1_pd(f1); }

    template <int N>
    Vec4 get1() const {
      return _mm256_set1_pd(arr[N]);  //FIXME
    }
    /*
    Vec4 get1(unsigned int n) const { 
      return _mm256_set1_pd(arr[n]); //FIXME
    }
    */
    double& operator[](unsigned int n) { return arr[n]; }

    double operator[](unsigned int n) const { return arr[n]; }

    Vec2<double> xy() const { return Vec2<double>(_mm256_castpd256_pd128(vec)); }
    Vec2<double> zw() const { return Vec2<double>(_mm256_castpd256_pd128(_mm256_permute2f128_pd(vec, vec, 1))); }
  };

  inline Vec4<float>::Vec4(Vec4<double> ivec) { vec = _mm256_cvtpd_ps(ivec.vec); }
}  // namespace mathSSE

inline bool operator==(mathSSE::Vec4<double> a, mathSSE::Vec4<double> b) {
  return _mm256_movemask_pd(_mm256_cmp_pd(a.vec, b.vec, _CMP_EQ_OS)) == 0xf;
}

inline mathSSE::Vec4<double> cmpeq(mathSSE::Vec4<double> a, mathSSE::Vec4<double> b) {
  return _mm256_cmp_pd(a.vec, b.vec, _CMP_EQ_OS);
}

inline mathSSE::Vec4<double> cmpgt(mathSSE::Vec4<double> a, mathSSE::Vec4<double> b) {
  return _mm256_cmp_pd(a.vec, b.vec, _CMP_GT_OS);
}

inline mathSSE::Vec4<double> hadd(mathSSE::Vec4<double> a, mathSSE::Vec4<double> b) {
  return _mm256_hadd_pd(a.vec, b.vec);
}

inline mathSSE::Vec4<double> operator-(mathSSE::Vec4<double> a) {
  const __m256d neg = _mm256_set_pd(-0.0, -0.0, -0.0, -0.0);
  return _mm256_xor_pd(a.vec, neg);
}

inline mathSSE::Vec4<double> operator&(mathSSE::Vec4<double> a, mathSSE::Vec4<double> b) {
  return _mm256_and_pd(a.vec, b.vec);
}
inline mathSSE::Vec4<double> operator|(mathSSE::Vec4<double> a, mathSSE::Vec4<double> b) {
  return _mm256_or_pd(a.vec, b.vec);
}
inline mathSSE::Vec4<double> operator^(mathSSE::Vec4<double> a, mathSSE::Vec4<double> b) {
  return _mm256_xor_pd(a.vec, b.vec);
}
inline mathSSE::Vec4<double> andnot(mathSSE::Vec4<double> a, mathSSE::Vec4<double> b) {
  return _mm256_andnot_pd(a.vec, b.vec);
}

inline mathSSE::Vec4<double> operator+(mathSSE::Vec4<double> a, mathSSE::Vec4<double> b) {
  return _mm256_add_pd(a.vec, b.vec);
}

inline mathSSE::Vec4<double> operator-(mathSSE::Vec4<double> a, mathSSE::Vec4<double> b) {
  return _mm256_sub_pd(a.vec, b.vec);
}

inline mathSSE::Vec4<double> operator*(mathSSE::Vec4<double> a, mathSSE::Vec4<double> b) {
  return _mm256_mul_pd(a.vec, b.vec);
}

inline mathSSE::Vec4<double> operator/(mathSSE::Vec4<double> a, mathSSE::Vec4<double> b) {
  return _mm256_div_pd(a.vec, b.vec);
}

inline mathSSE::Vec4<double> operator*(double a, mathSSE::Vec4<double> b) {
  return _mm256_mul_pd(_mm256_set1_pd(a), b.vec);
}

inline mathSSE::Vec4<double> operator*(mathSSE::Vec4<double> b, double a) {
  return _mm256_mul_pd(_mm256_set1_pd(a), b.vec);
}

inline mathSSE::Vec4<double> operator/(mathSSE::Vec4<double> b, double a) {
  return _mm256_div_pd(b.vec, _mm256_set1_pd(a));
}

inline double __attribute__((always_inline)) __attribute__((pure))
dot(mathSSE::Vec4<double> a, mathSSE::Vec4<double> b) {
  using mathSSE::_mm256_dot_pd;
  mathSSE::Vec4<double> ret;
  ret.vec = _mm256_dot_pd(a.vec, b.vec);
  return ret.arr[0];
}

inline mathSSE::Vec4<double> __attribute__((always_inline)) __attribute__((pure))
cross(mathSSE::Vec4<double> a, mathSSE::Vec4<double> b) {
  using mathSSE::_mm256_cross_pd;
  return _mm256_cross_pd(a.vec, b.vec);
}

inline double __attribute__((always_inline)) __attribute__((pure))
dotxy(mathSSE::Vec4<double> a, mathSSE::Vec4<double> b) {
  mathSSE::Vec4<double> mul = a * b;
  mul = hadd(mul, mul);
  return mul.arr[0];
}

#endif

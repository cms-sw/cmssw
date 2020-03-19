#ifndef PhysicsTools_Utilities_Simplify_begin_h
#define PhysicsTools_Utilities_Simplify_begin_h
#undef PhysicsTools_Utilities_Simplify_end_h
#define TYP0

#define TYPT1 typename A
#define TYPT2 typename A, typename B
#define TYPT3 typename A, typename B, typename C
#define TYPT4 typename A, typename B, typename C, typename D

#define TYPN1 int n
#define TYPN2 int n, int m
#define TYPN3 int n, int m, int k
#define TYPN4 int n, int m, int k, int l

#define TYPN1T1 int n, typename A
#define TYPN1T2 int n, typename A, typename B
#define TYPN2T1 int n, int m, typename A
#define TYPN2T2 int n, int m, typename A, typename B
#define TYPN3T1 int n, int m, int k, typename A

#define TYPX typename X

#define TYPXT1 typename X, typename A
#define TYPXT2 typename X, typename A, typename B

#define TYPXN1 typename X, int n
#define TYPXN2 typename X, int n, int m

#define TYPXN1T1 typename X, int n, typename A
#define TYPXN2T1 typename X, int n, int m, typename A

#define TEMPL(X) template <TYP##X>

#define NUM(N) Numerical<N>
#define FRACT(N, M) typename Fraction<N, M>::type
#define SUM_S(A, B) SumStruct<A, B>
#define MINUS_S(A) MinusStruct<A>
#define PROD_S(A, B) ProductStruct<A, B>
#define RATIO_S(A, B) RatioStruct<A, B>
#define POWER_S(A, B) PowerStruct<A, B>
#define FRACT_S(N, M) FractionStruct<N, M>
#define SQRT_S(A) SqrtStruct<A>
#define EXP_S(A) ExpStruct<A>
#define LOG_S(A) LogStruct<A>
#define SIN_S(A) SinStruct<A>
#define COS_S(A) CosStruct<A>
#define TAN_S(A) TanStruct<A>
#define ABS_S(A) AbsStruct<A>
#define SGN_S(A) SgnStruct<A>

#define SUM(A, B) typename Sum<A, B>::type
#define DIFF(A, B) typename Difference<A, B>::type
#define MINUS(A) typename Minus<A>::type
#define PROD(A, B) typename Product<A, B>::type
#define RATIO(A, B) typename Ratio<A, B>::type
#define POWER(A, B) typename Power<A, B>::type
#define SQUARE(A) typename Square<A>::type
#define SQRT(A) typename Sqrt<A>::type
#define EXP(A) typename Exp<A>::type
#define LOG(A) typename Log<A>::type
#define SIN(A) typename Sin<A>::type
#define COS(A) typename Cos<A>::type
#define SIN2(A) typename Sin2<A>::type
#define COS2(A) typename Cos2<A>::type
#define TAN(A) typename Tan<A>::type
#define ABS(A) typename Abs<A>::type
#define SGN(A) typename Sgn<A>::type

#define COMBINE(A, B, RES)                                             \
  inline static type combine(const A& _1, const B& _2) { return RES; } \
  struct __useless_ignoreme

#define MINUS_RULE(TMPL, T, RES, COMB)                      \
  template <TMPL>                                           \
  struct Minus<T> {                                         \
    typedef RES type;                                       \
    inline static type operate(const T& _) { return COMB; } \
  }

#define SUM_RULE(TMPL, T1, T2, RES, COMB)                                   \
  template <TMPL>                                                           \
  struct Sum<T1, T2> {                                                      \
    typedef RES type;                                                       \
    inline static type combine(const T1& _1, const T2& _2) { return COMB; } \
  }

#define DIFF_RULE(TMPL, T1, T2, RES, COMB)                                  \
  template <TMPL>                                                           \
  struct Difference<T1, T2> {                                               \
    typedef RES type;                                                       \
    inline static type combine(const T1& _1, const T2& _2) { return COMB; } \
  }

#define PROD_RULE(TMPL, T1, T2, RES, COMB)                                  \
  template <TMPL>                                                           \
  struct Product<T1, T2> {                                                  \
    typedef RES type;                                                       \
    inline static type combine(const T1& _1, const T2& _2) { return COMB; } \
  }

#define RATIO_RULE(TMPL, T1, T2, RES, COMB)                                 \
  template <TMPL>                                                           \
  struct Ratio<T1, T2> {                                                    \
    typedef RES type;                                                       \
    inline static type combine(const T1& _1, const T2& _2) { return COMB; } \
  }

#define POWER_RULE(TMPL, T1, T2, RES, COMB)                                 \
  template <TMPL>                                                           \
  struct Power<T1, T2> {                                                    \
    typedef RES type;                                                       \
    inline static type combine(const T1& _1, const T2& _2) { return COMB; } \
  }

#define EXP_RULE(TMPL, T, RES, COMB)                        \
  template <TMPL>                                           \
  struct Exp<T> {                                           \
    typedef RES type;                                       \
    inline static type compose(const T& _) { return COMB; } \
  }

#define LOG_RULE(TMPL, T, RES, COMB)                        \
  template <TMPL>                                           \
  struct Log<T> {                                           \
    typedef RES type;                                       \
    inline static type compose(const T& _) { return COMB; } \
  }

#define SIN_RULE(TMPL, T, RES, COMB)                        \
  template <TMPL>                                           \
  struct Sin<T> {                                           \
    typedef RES type;                                       \
    inline static type compose(const T& _) { return COMB; } \
  }

#define COS_RULE(TMPL, T, RES, COMB)                        \
  template <TMPL>                                           \
  struct Cos<T> {                                           \
    typedef RES type;                                       \
    inline static type compose(const T& _) { return COMB; } \
  }

#define TAN_RULE(TMPL, T, RES, COMB)                        \
  template <TMPL>                                           \
  struct Tan<T> {                                           \
    typedef RES type;                                       \
    inline static type compose(const T& _) { return COMB; } \
  }

#define GET(A, RES)                                  \
  inline static type get(const A& _) { return RES; } \
                                                     \
  struct __useless_ignoreme

#define DERIV(X, A) typename Derivative<X, A>::type
#define PRIMIT(X, A) typename Primitive<A, X>::type

#define DERIV_RULE(TMPL, T, RES, COMB)                  \
  template <TMPL>                                       \
  struct Derivative<X, T> {                             \
    typedef RES type;                                   \
    inline static type get(const T& _) { return COMB; } \
  }

#define PRIMIT_RULE(TMPL, T, RES, COMB)                 \
  template <TMPL>                                       \
  struct Primitive<T, X> {                              \
    typedef RES type;                                   \
    inline static type get(const T& _) { return COMB; } \
  }

#endif

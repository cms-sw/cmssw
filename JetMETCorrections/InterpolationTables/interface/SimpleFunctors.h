#ifndef NPSTAT_SIMPLEFUNCTORS_HH_
#define NPSTAT_SIMPLEFUNCTORS_HH_

/*!
// \file SimpleFunctors.h
//
// \brief Interface definitions and concrete simple functors for
//        a variety of functor-based calculations
//
// Author: I. Volobouev
//
// March 2009
*/

namespace npstat {
    /** Base class for a functor that takes no arguments */
    template <typename Result>
    struct Functor0
    {
        typedef Result result_type;

        inline virtual ~Functor0() {}
        virtual Result operator()() const = 0;
    };

    /** Base class for a functor that takes a single argument */
    template <typename Result, typename Arg1>
    struct Functor1
    {
        typedef Result result_type;
        typedef Arg1 first_argument_type;

        inline virtual ~Functor1() {}
        virtual Result operator()(const Arg1&) const = 0;
    };

    /** Base class for a functor that takes two arguments */
    template <typename Result, typename Arg1, typename Arg2>
    struct Functor2
    {
        typedef Result result_type;
        typedef Arg1 first_argument_type;
        typedef Arg2 second_argument_type;

        inline virtual ~Functor2() {}
        virtual Result operator()(const Arg1&, const Arg2&) const = 0;
    };

    /** Base class for a functor that takes three arguments */
    template <typename Result, typename Arg1, typename Arg2, typename Arg3>
    struct Functor3
    {
        typedef Result result_type;
        typedef Arg1 first_argument_type;
        typedef Arg2 second_argument_type;
        typedef Arg3 third_argument_type;

        inline virtual ~Functor3() {}
        virtual Result operator()(const Arg1&,const Arg2&,const Arg3&) const=0;
    };

    /** A simple functor which returns a copy of its argument */
    template <typename Result>
    struct Same : public Functor1<Result, Result>
    {
        inline Result operator()(const Result& a) const {return a;}
    };

    /** A simple functor which returns a reference to its argument */
    template <typename Result>
    struct SameRef : public Functor1<const Result&, Result>
    {
        inline const Result& operator()(const Result& a) const {return a;}
    };

    /**
    // Simple functor which ignores is arguments and instead
    // builds the result using the default constructor of the result type
    */
    template <typename Result>
    struct DefaultConstructor0 : public Functor0<Result>
    {
        inline Result operator()() const {return Result();}
    };

    /**
    // Simple functor which ignores is arguments and instead
    // builds the result using the default constructor of the result type
    */
    template <typename Result, typename Arg1>
    struct DefaultConstructor1 : public Functor1<Result, Arg1>
    {
        inline Result operator()(const Arg1&) const {return Result();}
    };

    /**
    // Simple functor which ignores is arguments and instead
    // builds the result using the default constructor of the result type
    */
    template <typename Result, typename Arg1, typename Arg2>
    struct DefaultConstructor2 : public Functor2<Result, Arg1, Arg2>
    {
        inline Result operator()(const Arg1&, const Arg2&) const
            {return Result();}
    };

    /**
    // Simple functor which ignores is arguments and instead
    // builds the result using the default constructor of the result type
    */
    template <typename Result, typename Arg1, typename Arg2, typename Arg3>
    struct DefaultConstructor3 : public Functor3<Result, Arg1, Arg2, Arg3>
    {
        inline Result operator()(const Arg1&, const Arg2&, const Arg3&) const
            {return Result();}
    };

    /**
    // Sometimes it becomes necessary to perform an explicit cast for proper
    // overload resolution of a converting copy constructor
    */
    template <typename Result, typename Arg1, typename CastType>
    struct CastingCopyConstructor : public Functor1<Result, Arg1>
    {
        inline Result operator()(const Arg1& a) const
            {return Result(static_cast<CastType>(a));}
    };

    /**
    // Adaptation for using functors without arguments with simple
    // cmath-style functions. Do not use this struct as a base class.
    */
    template <typename Result>
    struct FcnFunctor0 : public Functor0<Result>
    {
        inline explicit FcnFunctor0(Result (*fcn)()) : fcn_(fcn) {}

        inline Result operator()() const {return fcn_();}

    private:
        FcnFunctor0();
        Result (*fcn_)();
    };

    /**
    // Adaptation for using single-argument functors with simple
    // cmath-style functions. Do not use this struct as a base class.
    */
    template <typename Result, typename Arg1>
    struct FcnFunctor1 : public Functor1<Result, Arg1>
    {
        inline explicit FcnFunctor1(Result (*fcn)(Arg1)) : fcn_(fcn) {}

        inline Result operator()(const Arg1& a) const {return fcn_(a);}

    private:
        FcnFunctor1();
        Result (*fcn_)(Arg1);
    };

    /**
    // Adaptation for using two-argument functors with simple
    // cmath-style functions. Do not use this struct as a base class.
    */
    template <typename Result, typename Arg1, typename Arg2>
    struct FcnFunctor2 : public Functor2<Result, Arg1, Arg2>
    {
        inline explicit FcnFunctor2(Result (*fcn)(Arg1, Arg2)) : fcn_(fcn) {}

        inline Result operator()(const Arg1& x, const Arg2& y) const
            {return fcn_(x, y);}

    private:
        FcnFunctor2();
        Result (*fcn_)(Arg1, Arg2);
    };

    /**
    // Adaptation for using three-argument functors with simple
    // cmath-style functions. Do not use this struct as a base class.
    */
    template <typename Result, typename Arg1, typename Arg2, typename Arg3>
    struct FcnFunctor3 : public Functor3<Result, Arg1, Arg2, Arg3>
    {
        inline explicit FcnFunctor3(Result (*fcn)(Arg1,Arg2,Arg3)):fcn_(fcn) {}

        inline Result operator()(const Arg1&x,const Arg2&y,const Arg3&z) const
            {return fcn_(x, y, z);}

    private:
        FcnFunctor3();
        Result (*fcn_)(Arg1, Arg2, Arg3);
    };

    /**
    // Functor which extracts a given element from a random access linear
    // container without bounds checking
    */
    template <class Container, class Result = typename Container::value_type>
    struct Element1D : public Functor1<Result, Container>
    {
        inline explicit Element1D(const unsigned long index) : idx(index) {}

        inline Result operator()(const Container& c) const {return c[idx];}

    private:
        Element1D();
        unsigned long idx;
    };

    /**
    // Functor which extracts a given element from a random access linear
    // container with bounds checking
    */
    template <class Container, class Result = typename Container::value_type>
    struct Element1DAt : public Functor1<Result, Container>
    {
        inline explicit Element1DAt(const unsigned long index) : idx(index) {}

        inline Result operator()(const Container& c) const {return c.at(idx);}

    private:
        Element1DAt();
        unsigned long idx;
    };

    /** 
    // Left assignment functor. Works just like normal binary
    // assignment operator in places where functor is needed.
    */
    template <typename T1, typename T2>
    struct assign_left
    {
        inline T1& operator()(T1& left, const T2& right) const
            {return left = right;}
    };

    /** 
    // Right assignment functor. Reverses the assignment direction
    // in comparison with the normal binary assignment operator.
    */
    template <typename T1, typename T2>
    struct assign_right
    {
        inline T2& operator()(const T1& left, T2& right) const
            {return right = left;}
    };

    /** In-place addition on the left side */
    template <typename T1, typename T2>
    struct pluseq_left
    {
        inline T1& operator()(T1& left, const T2& right) const
            {return left += right;}
    };

    /** In-place addition on the right side */
    template <typename T1, typename T2>
    struct pluseq_right
    {
        inline T2& operator()(const T1& left, T2& right) const
            {return right += left;}
    };

    /** 
    // In-place addition on the left side preceded by
    // multiplication of the right argument by a double
    */
    template <typename T1, typename T2>
    struct addmul_left
    {
        inline explicit addmul_left(const double weight) : w_(weight) {}

        inline T1& operator()(T1& left, const T2& right) const
            {return left += w_*right;}

    private:
        addmul_left();
        double w_;
    };

    /** 
    // In-place addition on the right side preceded by
    // multiplication of the left argument by a double
    */
    template <typename T1, typename T2>
    struct addmul_right
    {
        inline explicit addmul_right(const double weight) : w_(weight) {}

        inline T1& operator()(T1& left, const T2& right) const
            {return right += w_*left;}

    private:
        addmul_right();
        double w_;
    };

    /** In-place subtraction on the left side */
    template <typename T1, typename T2>
    struct minuseq_left
    {
        inline T1& operator()(T1& left, const T2& right) const
            {return left -= right;}
    };

    /** In-place subtraction on the right side */
    template <typename T1, typename T2>
    struct minuseq_right
    {
        inline T2& operator()(const T1& left, T2& right) const
            {return right -= left;}
    };

    /** In-place multiplication on the left side */
    template <typename T1, typename T2>
    struct multeq_left
    {
        inline T1& operator()(T1& left, const T2& right) const
            {return left *= right;}
    };

    /** In-place multiplication on the right side */
    template <typename T1, typename T2>
    struct multeq_right
    {
        inline T2& operator()(const T1& left, T2& right) const
            {return right *= left;}
    };

    /** In-place division on the left side withot checking for division by 0 */
    template <typename T1, typename T2>
    struct diveq_left
    {
        inline T1& operator()(T1& left, const T2& right) const
            {return left /= right;}
    };

    /** In-place division on the right side withot checking for division by 0 */
    template <typename T1, typename T2>
    struct diveq_right
    {
        inline T2& operator()(const T1& left, T2& right) const
            {return right /= left;}
    };

    /** In-place division on the left side. Allow 0/0 = const. */
    template <typename T1, typename T2>
    struct diveq_left_0by0isC
    {
        inline diveq_left_0by0isC() : 
            C(T1()), leftZero(T1()), rightZero(T2()) {}
        inline explicit diveq_left_0by0isC(const T1& value) :
            C(value), leftZero(T1()), rightZero(T2()) {}

        inline T1& operator()(T1& left, const T2& right) const
        {
            if (right == rightZero)
                if (left == leftZero)
                {
                    left = C;
                    return left;
                }
            return left /= right;
        }

    private:
        T1 C;
        T1 leftZero;
        T2 rightZero;
    };

    /** In-place division on the right side. Allow 0/0 = const. */
    template <typename T1, typename T2>
    struct diveq_right_0by0isC
    {
        inline diveq_right_0by0isC() :
            C(T2()), leftZero(T1()), rightZero(T2())  {}
        inline explicit diveq_right_0by0isC(const T2& value) :
            C(value), leftZero(T1()), rightZero(T2()) {}

        inline T2& operator()(const T1& left, T2& right) const
        {
            if (left == leftZero)
                if (right == rightZero)
                {
                    right = C;
                    return right;
                }
            return right /= left;
        }

    private:
        T2 C;
        T1 leftZero;
        T2 rightZero;
    };

    /** Left assignment functor preceded by a static cast */
    template <typename T1, typename T2, typename T3=T1>
    struct scast_assign_left
    {
        inline T1& operator()(T1& left, const T2& right) const
            {return left = static_cast<T3>(right);}
    };

    /** Right assignment functor preceded by a static cast */
    template <typename T1, typename T2, typename T3=T2>
    struct scast_assign_right
    {
        inline T2& operator()(const T1& left, T2& right) const
            {return right = static_cast<T3>(left);}
    };

    /** In-place addition on the left side preceded by a static cast */
    template <typename T1, typename T2, typename T3=T1>
    struct scast_pluseq_left
    {
        inline T1& operator()(T1& left, const T2& right) const
            {return left += static_cast<T3>(right);}
    };

    /** In-place addition on the right side preceded by a static cast */
    template <typename T1, typename T2, typename T3=T2>
    struct scast_pluseq_right
    {
        inline T2& operator()(const T1& left, T2& right) const
            {return right += static_cast<T3>(left);}
    };

    /** In-place subtraction on the left side preceded by a static cast */
    template <typename T1, typename T2, typename T3=T1>
    struct scast_minuseq_left
    {
        inline T1& operator()(T1& left, const T2& right) const
            {return left -= static_cast<T3>(right);}
    };

    /** In-place subtraction on the right side preceded by a static cast */
    template <typename T1, typename T2, typename T3=T2>
    struct scast_minuseq_right
    {
        inline T2& operator()(const T1& left, T2& right) const
            {return right -= static_cast<T3>(left);}
    };
}

#endif // NPSTAT_SIMPLEFUNCTORS_HH_


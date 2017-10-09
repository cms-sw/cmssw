#ifndef RecoJets_FFTJetAlgorithms_ScaleCalculators_h
#define RecoJets_FFTJetAlgorithms_ScaleCalculators_h

#include <vector>
#include <algorithm>

#include "fftjet/SimpleFunctors.hh"
#include "fftjet/RecombinedJet.hh"

#include "RecoJets/FFTJetAlgorithms/interface/fftjetTypedefs.h"

namespace fftjetcms {
    // Return a predefined constant
    template <typename Arg1>
    class ConstDouble : public fftjet::Functor1<double,Arg1>
    {
    public:
        inline ConstDouble(const double value) : c_(value) {}
        inline double operator()(const Arg1&) const override {return c_;}

    private:
        ConstDouble() = delete;
        double c_;
    };


    // Multiplication of a constant by the "scale()" member
    template <class T>
    class ProportionalToScale : public fftjet::Functor1<double,T>
    {
    public:
        inline ProportionalToScale(const double value) : c_(value) {}
        inline double operator()(const T& r) const override {return r.scale()*c_;}

    private:
        ProportionalToScale() = delete;
        double c_;
    };


    // Multiplication by a constant
    template <class T>
    class MultiplyByConst : public fftjet::Functor1<double,T>
    {
    public:
        inline MultiplyByConst(const double factor,
                               const fftjet::Functor1<double,T>* f,
                               const bool takeOwnership=false)
            : c_(factor), func_(f), ownsPointer_(takeOwnership) {}

        inline ~MultiplyByConst() override {if (ownsPointer_) delete func_;}

        inline double operator()(const T& r) const override {return (*func_)(r)*c_;}

    private:
        MultiplyByConst() = delete;
        double c_;
        const fftjet::Functor1<double,T>* func_;
        const bool ownsPointer_;
    };


    // Function composition
    template <class T>
    class CompositeFunctor : public fftjet::Functor1<double,T>
    {
    public:
        inline CompositeFunctor(const fftjet::Functor1<double,double>* f1,
                                const fftjet::Functor1<double,T>* f2,
                                const bool takeOwnership=false)
            : f1_(f1), f2_(f2), ownsPointers_(takeOwnership) {}

        inline ~CompositeFunctor() override
            {if (ownsPointers_) {delete f1_; delete f2_;}}

        inline double operator()(const T& r) const override {return (*f1_)((*f2_)(r));}

    private:
        CompositeFunctor() = delete;
        const fftjet::Functor1<double,double>* f1_;
        const fftjet::Functor1<double,T>* f2_;
        const bool ownsPointers_;
    };


    // Product of two functors
    template <class T>
    class ProductFunctor : public fftjet::Functor1<double,T>
    {
    public:
        inline ProductFunctor(const fftjet::Functor1<double,T>* f1,
                              const fftjet::Functor1<double,T>* f2,
                              const bool takeOwnership=false)
            : f1_(f1), f2_(f2), ownsPointers_(takeOwnership) {}

        inline ~ProductFunctor() override
            {if (ownsPointers_) {delete f1_; delete f2_;}}

        inline double operator()(const T& r) const override
            {return (*f1_)(r) * (*f2_)(r);}

    private:
        ProductFunctor() = delete;
        const fftjet::Functor1<double,T>* f1_;
        const fftjet::Functor1<double,T>* f2_;
        const bool ownsPointers_;
    };


    // Function dependent on magnitude
    template <class T>
    class MagnitudeDependent : public fftjet::Functor1<double,T>
    {
    public:
        inline MagnitudeDependent(const fftjet::Functor1<double,double>* f1,
                                  const bool takeOwnership=false)
            : f1_(f1), ownsPointer_(takeOwnership) {}

        inline ~MagnitudeDependent() override {if (ownsPointer_) delete f1_;}

        inline double operator()(const T& r) const override
            {return (*f1_)(r.magnitude());}

    private:
        MagnitudeDependent() = delete;
        const fftjet::Functor1<double,double>* f1_;
        const bool ownsPointer_;
    };


    // Functions dependent on peak eta
    class PeakEtaDependent : public fftjet::Functor1<double,fftjet::Peak>
    {
    public:
        inline PeakEtaDependent(const fftjet::Functor1<double,double>* f1,
                                const bool takeOwnership=false)
            : f1_(f1), ownsPointer_(takeOwnership) {}

        inline ~PeakEtaDependent() override {if (ownsPointer_) delete f1_;}

        inline double operator()(const fftjet::Peak& r) const override
            {return (*f1_)(r.eta());}

    private:
        PeakEtaDependent() = delete;
        const fftjet::Functor1<double,double>* f1_;
        const bool ownsPointer_;
    };


    class PeakEtaMagSsqDependent : public fftjet::Functor1<double,fftjet::Peak>
    {
    public:
        inline PeakEtaMagSsqDependent(
            const fftjet::LinearInterpolator2d* f1,
            const bool takeOwnership,
            const fftjet::JetMagnitudeMapper2d<fftjet::Peak>* jmmp,
            const bool ownjmp, const double fc)
            : f1_(f1), ownsPointer_(takeOwnership), jmmp_(jmmp),
              ownsjmmpPointer_(ownjmp), factor_(fc) {}

        inline ~PeakEtaMagSsqDependent() override
        {
            if (ownsPointer_) delete f1_;
            if (ownsjmmpPointer_) delete jmmp_;
        }

        inline double operator()(const fftjet::Peak& r) const override
        {
            const double scale = r.scale();
            const double magnitude = r.magnitude();
            const double pt = scale*scale*factor_*magnitude;
            const double partonpt = (*jmmp_)(pt,r);
            return (*f1_)(std::abs(r.eta()),partonpt)  ;
        }

    private:
        PeakEtaMagSsqDependent() = delete;
        const fftjet::LinearInterpolator2d* f1_;
        const bool ownsPointer_;
        const fftjet::JetMagnitudeMapper2d <fftjet::Peak>* jmmp_;
        const bool ownsjmmpPointer_;
        const double factor_;
    };


    // Functions dependent on jet eta
    class JetEtaDependent : 
        public fftjet::Functor1<double,fftjet::RecombinedJet<VectorLike> >
    {
    public:
        inline JetEtaDependent(const fftjet::Functor1<double,double>* f1,
                               const bool takeOwnership=false)
            : f1_(f1), ownsPointer_(takeOwnership) {}

        inline ~JetEtaDependent() override {if (ownsPointer_) delete f1_;}

        inline double operator()(
            const fftjet::RecombinedJet<VectorLike>& r) const override
            {return (*f1_)(r.vec().eta());}

    private:
        JetEtaDependent() = delete;
        const fftjet::Functor1<double,double>* f1_;
        const bool ownsPointer_;
    };


    // A simple polynomial. Coefficients are in the order c0, c1, c2, ...
    class Polynomial : public fftjet::Functor1<double,double>
    {
    public:
        inline Polynomial(const std::vector<double>& coeffs)
            : coeffs_(nullptr), nCoeffs(coeffs.size())
        {
            if (nCoeffs)
            {
                coeffs_ = new double[nCoeffs];
                std::copy(coeffs.begin(), coeffs.end(), coeffs_);
            }
        }
        inline ~Polynomial() override {delete [] coeffs_;}

        inline double operator()(const double& x) const override
        {
            double sum = 0.0;
            const double* p = coeffs_ + nCoeffs - 1;
            for (unsigned i=0; i<nCoeffs; ++i)
            {
                sum *= x;
                sum += *p--;
            }
            return sum;
        }

    private:
        Polynomial() = delete;
        double* coeffs_;
        const unsigned nCoeffs;
    };
}

#endif // RecoJets_FFTJetAlgorithms_ScaleCalculators_h

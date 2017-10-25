#ifndef RecoJets_FFTJetAlgorithms_JetAbsEta_h
#define RecoJets_FFTJetAlgorithms_JetAbsEta_h

#include <cmath>

#include "fftjet/SimpleFunctors.hh"

namespace fftjetcms {
    template<class Jet>
    struct PeakAbsEta : public fftjet::Functor1<double,Jet>
    {
        inline double operator()(const Jet& j) const override
            {return fabs(j.eta());}
    };

    template<class Jet>
    struct JetAbsEta : public fftjet::Functor1<double,Jet>
    {
        inline double operator()(const Jet& j) const override
            {return fabs(j.vec().Eta());}
    };
}

#endif // RecoJets_FFTJetAlgorithms_JetAbsEta_h

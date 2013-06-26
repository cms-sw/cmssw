#ifndef FFTJetCorrectorTransientFromJet_h
#define FFTJetCorrectorTransientFromJet_h

#include "JetMETCorrections/FFTJetObjects/interface/FFTJetCorrectorTransient.h"
#include "DataFormats/JetReco/interface/Jet.h"

template<typename MyJet>
struct FFTJetCorrectorTransientFromJet
{
    typedef MyJet jet_type;
    typedef FFTJetCorrectorTransient result_type;

    inline result_type operator()(const jet_type& j) const
        {return result_type(j.getFFTSpecific().f_vec());}
};

#endif // FFTJetCorrectorTransientFromJet_h

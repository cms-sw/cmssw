#ifndef FFTJetCorrectorResultFromTransient_h
#define FFTJetCorrectorResultFromTransient_h

#include "JetMETCorrections/FFTJetObjects/interface/FFTJetCorrectorResult.h"
#include "JetMETCorrections/FFTJetObjects/interface/FFTJetCorrectorTransient.h"
#include "DataFormats/JetReco/interface/Jet.h"

template<typename MyJet>
struct FFTJetCorrectorResultFromTransient
{
    typedef MyJet jet_type;
    typedef FFTJetCorrectorResult result_type;

    inline result_type operator()(const MyJet& /* jet */,
                                  const FFTJetCorrectorTransient& t) const
        {return FFTJetCorrectorResult(t.vec(), t.scale(), t.sigma());}
};

#endif // FFTJetCorrectorResultFromTransient_h

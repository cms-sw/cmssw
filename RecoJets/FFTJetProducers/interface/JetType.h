#ifndef RecoJets_FFTJetProducers_JetType_h
#define RecoJets_FFTJetProducers_JetType_h

#include <string>

namespace fftjetcms {
    enum JetType
    {
        BASICJET = 0,
        GENJET,
        CALOJET,
        PFJET,
        TRACKJET,
        JPTJET
    };

    JetType parseJetType(const std::string& name);
}

#endif // RecoJets_FFTJetProducers_JetType_h

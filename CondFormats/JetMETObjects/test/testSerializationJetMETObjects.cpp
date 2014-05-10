#include "CondFormats/Serialization/interface/Test.h"

#include "../src/headers.h"

int main()
{
    testSerialization<FFTJetCorrectorParameters>();
    //testSerialization<FactorizedJetCorrector>(); depends on non-serializable SimpleJetCorrector
    //testSerialization<JetCorrectionUncertainty>(); has pointers as members
    testSerialization<JetCorrectorParameters>();
    testSerialization<JetCorrectorParameters::Definitions>();
    testSerialization<JetCorrectorParameters::Record>();
    testSerialization<JetCorrectorParametersCollection>();
    testSerialization<JetCorrectorParametersCollection::collection_type>();
    testSerialization<JetCorrectorParametersCollection::pair_type>();
    //testSerialization<SimpleJetCorrectionUncertainty>(); has pointers as members
    //testSerialization<SimpleJetCorrector>(); has pointers as members
    testSerialization<std::vector<JetCorrectorParameters::Record>>();
    testSerialization<std::vector<JetCorrectorParameters>>();
    testSerialization<std::vector<JetCorrectorParametersCollection>>();
    testSerialization<QGLikelihoodCategory>();
    testSerialization<QGLikelihoodObject>();

    return 0;
}

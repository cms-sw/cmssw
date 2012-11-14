#ifndef JetMETCorrections_FFTJetObjects_FFTJetCorrectorParametersLoader_h
#define JetMETCorrections_FFTJetObjects_FFTJetCorrectorParametersLoader_h

#include "JetMETCorrections/FFTJetObjects/interface/FFTJetRcdMapper.h"
#include "CondFormats/JetMETObjects/interface/FFTJetCorrectorParameters.h"

//
// Note that the class below does not have any public constructors.
// All application usage is via the StaticFFTJetRcdMapper wrapper.
//
class FFTJetCorrectorParametersLoader :
    public DefaultFFTJetRcdMapper<FFTJetCorrectorParameters>
{
    typedef DefaultFFTJetRcdMapper<FFTJetCorrectorParameters> Base;
    friend class StaticFFTJetRcdMapper<FFTJetCorrectorParametersLoader>;
    FFTJetCorrectorParametersLoader();
};
        
typedef StaticFFTJetRcdMapper<FFTJetCorrectorParametersLoader>
StaticFFTJetCorrectorParametersLoader;

#endif // JetMETCorrections_FFTJetObjects_FFTJetCorrectorParametersLoader_h

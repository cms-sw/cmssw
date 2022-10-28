#ifndef JetMETCorrections_FFTJetObjects_FFTJetCorrectorParametersLoader_h
#define JetMETCorrections_FFTJetObjects_FFTJetCorrectorParametersLoader_h

#include "JetMETCorrections/FFTJetObjects/interface/FFTJetRcdMapper.h"
#include "CondFormats/JetMETObjects/interface/FFTJetCorrectorParameters.h"

struct FFTJetCorrectorParametersLoader : public DefaultFFTJetRcdMapper<FFTJetCorrectorParameters> {
  typedef DefaultFFTJetRcdMapper<FFTJetCorrectorParameters> Base;
  FFTJetCorrectorParametersLoader();
};

#endif  // JetMETCorrections_FFTJetObjects_FFTJetCorrectorParametersLoader_h

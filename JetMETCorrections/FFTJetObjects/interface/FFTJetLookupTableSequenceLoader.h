#ifndef JetMETCorrections_FFTJetObjects_FFTJetLookupTableSequenceLoader_h
#define JetMETCorrections_FFTJetObjects_FFTJetLookupTableSequenceLoader_h

#include "JetMETCorrections/FFTJetObjects/interface/FFTJetRcdMapper.h"
#include "JetMETCorrections/FFTJetObjects/interface/FFTJetLookupTableSequence.h"

struct FFTJetLookupTableSequenceLoader : public DefaultFFTJetRcdMapper<FFTJetLookupTableSequence> {
  typedef DefaultFFTJetRcdMapper<FFTJetLookupTableSequence> Base;
  FFTJetLookupTableSequenceLoader();
};

#endif  // JetMETCorrections_FFTJetObjects_FFTJetLookupTableSequenceLoader_h

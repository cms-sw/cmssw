#ifndef JetMETCorrections_FFTJetObjects_FFTJetCorrectorSequenceLoader_h
#define JetMETCorrections_FFTJetObjects_FFTJetCorrectorSequenceLoader_h

#include "JetMETCorrections/FFTJetObjects/interface/FFTJetRcdMapper.h"
#include "JetMETCorrections/FFTJetObjects/interface/FFTJetCorrectorSequenceTypes.h"

struct FFTBasicJetCorrectorSequenceLoader : public DefaultFFTJetRcdMapper<FFTBasicJetCorrectorSequence> {
  typedef DefaultFFTJetRcdMapper<FFTBasicJetCorrectorSequence> Base;
  FFTBasicJetCorrectorSequenceLoader();
};

struct FFTCaloJetCorrectorSequenceLoader : public DefaultFFTJetRcdMapper<FFTCaloJetCorrectorSequence> {
  typedef DefaultFFTJetRcdMapper<FFTCaloJetCorrectorSequence> Base;
  FFTCaloJetCorrectorSequenceLoader();
};

struct FFTGenJetCorrectorSequenceLoader : public DefaultFFTJetRcdMapper<FFTGenJetCorrectorSequence> {
  typedef DefaultFFTJetRcdMapper<FFTGenJetCorrectorSequence> Base;
  FFTGenJetCorrectorSequenceLoader();
};

struct FFTPFJetCorrectorSequenceLoader : public DefaultFFTJetRcdMapper<FFTPFJetCorrectorSequence> {
  typedef DefaultFFTJetRcdMapper<FFTPFJetCorrectorSequence> Base;
  FFTPFJetCorrectorSequenceLoader();
};

struct FFTTrackJetCorrectorSequenceLoader : public DefaultFFTJetRcdMapper<FFTTrackJetCorrectorSequence> {
  typedef DefaultFFTJetRcdMapper<FFTTrackJetCorrectorSequence> Base;
  FFTTrackJetCorrectorSequenceLoader();
};

struct FFTJPTJetCorrectorSequenceLoader : public DefaultFFTJetRcdMapper<FFTJPTJetCorrectorSequence> {
  typedef DefaultFFTJetRcdMapper<FFTJPTJetCorrectorSequence> Base;
  FFTJPTJetCorrectorSequenceLoader();
};

#endif  // JetMETCorrections_FFTJetObjects_FFTJetCorrectorSequenceLoader_h

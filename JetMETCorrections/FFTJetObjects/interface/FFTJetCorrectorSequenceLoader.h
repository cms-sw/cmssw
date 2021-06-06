#ifndef JetMETCorrections_FFTJetObjects_FFTJetCorrectorSequenceLoader_h
#define JetMETCorrections_FFTJetObjects_FFTJetCorrectorSequenceLoader_h

#include "JetMETCorrections/FFTJetObjects/interface/FFTJetRcdMapper.h"
#include "JetMETCorrections/FFTJetObjects/interface/FFTJetCorrectorSequenceTypes.h"

class FFTBasicJetCorrectorSequenceLoader : public DefaultFFTJetRcdMapper<FFTBasicJetCorrectorSequence> {
  typedef DefaultFFTJetRcdMapper<FFTBasicJetCorrectorSequence> Base;
  friend class StaticFFTJetRcdMapper<FFTBasicJetCorrectorSequenceLoader>;
  FFTBasicJetCorrectorSequenceLoader();  // NOLINT - prevent clang-tidy from adding `= delete`
};

typedef StaticFFTJetRcdMapper<FFTBasicJetCorrectorSequenceLoader> StaticFFTBasicJetCorrectorSequenceLoader;

class FFTCaloJetCorrectorSequenceLoader : public DefaultFFTJetRcdMapper<FFTCaloJetCorrectorSequence> {
  typedef DefaultFFTJetRcdMapper<FFTCaloJetCorrectorSequence> Base;
  friend class StaticFFTJetRcdMapper<FFTCaloJetCorrectorSequenceLoader>;
  FFTCaloJetCorrectorSequenceLoader();  // NOLINT - prevent clang-tidy from adding `= delete`
};

typedef StaticFFTJetRcdMapper<FFTCaloJetCorrectorSequenceLoader> StaticFFTCaloJetCorrectorSequenceLoader;

class FFTGenJetCorrectorSequenceLoader : public DefaultFFTJetRcdMapper<FFTGenJetCorrectorSequence> {
  typedef DefaultFFTJetRcdMapper<FFTGenJetCorrectorSequence> Base;
  friend class StaticFFTJetRcdMapper<FFTGenJetCorrectorSequenceLoader>;
  FFTGenJetCorrectorSequenceLoader();  // NOLINT - prevent clang-tidy from adding `= delete`
};

typedef StaticFFTJetRcdMapper<FFTGenJetCorrectorSequenceLoader> StaticFFTGenJetCorrectorSequenceLoader;

class FFTPFJetCorrectorSequenceLoader : public DefaultFFTJetRcdMapper<FFTPFJetCorrectorSequence> {
  typedef DefaultFFTJetRcdMapper<FFTPFJetCorrectorSequence> Base;
  friend class StaticFFTJetRcdMapper<FFTPFJetCorrectorSequenceLoader>;
  FFTPFJetCorrectorSequenceLoader();  // NOLINT - prevent clang-tidy from adding `= delete`
};

typedef StaticFFTJetRcdMapper<FFTPFJetCorrectorSequenceLoader> StaticFFTPFJetCorrectorSequenceLoader;

class FFTTrackJetCorrectorSequenceLoader : public DefaultFFTJetRcdMapper<FFTTrackJetCorrectorSequence> {
  typedef DefaultFFTJetRcdMapper<FFTTrackJetCorrectorSequence> Base;
  friend class StaticFFTJetRcdMapper<FFTTrackJetCorrectorSequenceLoader>;
  FFTTrackJetCorrectorSequenceLoader();  // NOLINT - prevent clang-tidy from adding `= delete`
};

typedef StaticFFTJetRcdMapper<FFTTrackJetCorrectorSequenceLoader> StaticFFTTrackJetCorrectorSequenceLoader;

class FFTJPTJetCorrectorSequenceLoader : public DefaultFFTJetRcdMapper<FFTJPTJetCorrectorSequence> {
  typedef DefaultFFTJetRcdMapper<FFTJPTJetCorrectorSequence> Base;
  friend class StaticFFTJetRcdMapper<FFTJPTJetCorrectorSequenceLoader>;
  FFTJPTJetCorrectorSequenceLoader();  // NOLINT - prevent clang-tidy from adding `= delete`
};

typedef StaticFFTJetRcdMapper<FFTJPTJetCorrectorSequenceLoader> StaticFFTJPTJetCorrectorSequenceLoader;

#endif  // JetMETCorrections_FFTJetObjects_FFTJetCorrectorSequenceLoader_h

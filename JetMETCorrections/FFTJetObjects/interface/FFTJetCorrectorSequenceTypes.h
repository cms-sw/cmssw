#ifndef JetMETCorrections_FFTJetObjects_FFTJetCorrectorSequenceTypes_h
#define JetMETCorrections_FFTJetObjects_FFTJetCorrectorSequenceTypes_h

#include "JetMETCorrections/FFTJetObjects/interface/FFTJetCorrectorSequence.h"
#include "JetMETCorrections/FFTJetObjects/interface/FFTJetCorrectorTransientFromJet.h"
#include "JetMETCorrections/FFTJetObjects/interface/FFTJetCorrectorResultFromTransient.h"
#include "JetMETCorrections/FFTJetObjects/interface/FFTJetCorrectionsTypemap.h"

typedef FFTJetCorrectorSequence<
    reco::FFTAnyJet<reco::BasicJet>,
    FFTJetCorrectorTransientFromJet,
    FFTJetCorrectorResultFromTransient
> FFTBasicJetCorrectorSequence;

typedef FFTJetCorrectorSequence<
    reco::FFTAnyJet<reco::CaloJet>,
    FFTJetCorrectorTransientFromJet,
    FFTJetCorrectorResultFromTransient
> FFTCaloJetCorrectorSequence;

typedef FFTJetCorrectorSequence<
    reco::FFTAnyJet<reco::GenJet>,
    FFTJetCorrectorTransientFromJet,
    FFTJetCorrectorResultFromTransient
> FFTGenJetCorrectorSequence;

typedef FFTJetCorrectorSequence<
    reco::FFTAnyJet<reco::PFJet>,
    FFTJetCorrectorTransientFromJet,
    FFTJetCorrectorResultFromTransient
> FFTPFJetCorrectorSequence;

typedef FFTJetCorrectorSequence<
    reco::FFTAnyJet<reco::TrackJet>,
    FFTJetCorrectorTransientFromJet,
    FFTJetCorrectorResultFromTransient
> FFTTrackJetCorrectorSequence;

typedef FFTJetCorrectorSequence<
    reco::FFTAnyJet<reco::JPTJet>,
    FFTJetCorrectorTransientFromJet,
    FFTJetCorrectorResultFromTransient
> FFTJPTJetCorrectorSequence;

#endif // JetMETCorrections_FFTJetObjects_FFTJetCorrectorSequenceTypes_h

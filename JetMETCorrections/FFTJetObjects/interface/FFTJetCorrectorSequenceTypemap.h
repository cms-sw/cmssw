#ifndef JetMETCorrections_FFTJetObjects_FFTJetCorrectorSequenceTypemap_h
#define JetMETCorrections_FFTJetObjects_FFTJetCorrectorSequenceTypemap_h

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/JPTJet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/BasicJet.h"
#include "DataFormats/JetReco/interface/TrackJet.h"
#include "DataFormats/JetReco/interface/FFTAnyJet.h"

#include "JetMETCorrections/FFTJetObjects/interface/FFTJetCorrectorSequenceLoader.h"

template <typename T>
struct FFTJetCorrectorSequenceTypemap {};

template <>
struct FFTJetCorrectorSequenceTypemap<reco::FFTAnyJet<reco::BasicJet> > {
  typedef FFTBasicJetCorrectorSequenceLoader loader;
};

template <>
struct FFTJetCorrectorSequenceTypemap<reco::FFTAnyJet<reco::CaloJet> > {
  typedef FFTCaloJetCorrectorSequenceLoader loader;
};

template <>
struct FFTJetCorrectorSequenceTypemap<reco::FFTAnyJet<reco::GenJet> > {
  typedef FFTGenJetCorrectorSequenceLoader loader;
};

template <>
struct FFTJetCorrectorSequenceTypemap<reco::FFTAnyJet<reco::PFJet> > {
  typedef FFTPFJetCorrectorSequenceLoader loader;
};

template <>
struct FFTJetCorrectorSequenceTypemap<reco::FFTAnyJet<reco::TrackJet> > {
  typedef FFTTrackJetCorrectorSequenceLoader loader;
};

template <>
struct FFTJetCorrectorSequenceTypemap<reco::FFTAnyJet<reco::JPTJet> > {
  typedef FFTJPTJetCorrectorSequenceLoader loader;
};

#endif  // JetMETCorrections_FFTJetObjects_FFTJetCorrectorSequenceTypemap_h

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

template<typename T>
struct FFTJetCorrectorSequenceTypemap {};

template<>
struct FFTJetCorrectorSequenceTypemap<reco::FFTAnyJet<reco::BasicJet> >
{typedef StaticFFTBasicJetCorrectorSequenceLoader loader;};

template<>
struct FFTJetCorrectorSequenceTypemap<reco::FFTAnyJet<reco::CaloJet> >
{typedef StaticFFTCaloJetCorrectorSequenceLoader loader;};

template<>
struct FFTJetCorrectorSequenceTypemap<reco::FFTAnyJet<reco::GenJet> >
{typedef StaticFFTGenJetCorrectorSequenceLoader loader;};
 
template<>
struct FFTJetCorrectorSequenceTypemap<reco::FFTAnyJet<reco::PFJet> >
{typedef StaticFFTPFJetCorrectorSequenceLoader loader;};

template<>
struct FFTJetCorrectorSequenceTypemap<reco::FFTAnyJet<reco::TrackJet> >
{typedef StaticFFTTrackJetCorrectorSequenceLoader loader;};

template<>
struct FFTJetCorrectorSequenceTypemap<reco::FFTAnyJet<reco::JPTJet> >
{typedef StaticFFTJPTJetCorrectorSequenceLoader loader;};

#endif // JetMETCorrections_FFTJetObjects_FFTJetCorrectorSequenceTypemap_h

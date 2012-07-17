#
# FFTJet simple jet analyzer configuration
#
# I. Volobouev, July 16, 2012
#
import math
import FWCore.ParameterSet.Config as cms

fftjetSimpleAnalyzer = cms.EDAnalyzer(
    "SimpleFFTJetAnalyzer",
    #
    # Which info we are going to collect?
    collectGenJets = cms.bool(False),
    collectCaloJets = cms.bool(False),
    collectPFJets = cms.bool(True),
    #
    # Labels for jet collections and FFTJetProducer summaries
    genjetCollectionLabel = cms.InputTag("fftjetproducer", "MadeByFFTJet"),
    caloCollectionLabel = cms.InputTag("fftjetproducer", "MadeByFFTJet"),
    pfCollectionLabel = cms.InputTag("fftjetproducer", "MadeByFFTJet"),
    #
    # Ntuple names and titles. Should be unique for the whole job.
    genjetTitle = cms.string("FFT_GenJets"),
    caloTitle = cms.string("FFT_CaloJets"),
    pfTitle = cms.string("FFT_PFJets"),
    #
    # Conversion factor from scale squared times peak magnitude to Pt.
    # Note that this factor depends on the grid used for pattern resolution.
    # The default value given here is correct for the default grid only.
    ptConversionFactor = cms.double(128*256/(4.0*math.pi))
)

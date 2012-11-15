import FWCore.ParameterSet.Config as cms
from JetMETCorrections.FFTJetModules.fftjetcorrectionesproducer_cfi import *

def configure_fftjet_correction_producer(sequenceTags, jetProducerModule):
    used_records = [fftjet_corr_types[s].correctorRecord for s in sequenceTags]
    config = cms.EDProducer(
        "FFTJetCorrectionProducer",
        #
        # Input jet collection
        src = cms.InputTag(jetProducerModule, "MadeByFFTJet"),
        #
        # Label for the output jet collection
        outputLabel = cms.string(""),
        #
        # Event setup record types for jet correction sequences
        records = cms.vstring(used_records),
        #
        # Jet type to process
        jetType = cms.string(fftjet_corr_types[sequenceTags[0]].jetType),
        #
        # Are we going to write out a corresponding collection
        # of systematic uncertainties?
        writeUncertainties = cms.bool(False),
        #
        # Subtract the pileup?
        subtractPileup = cms.bool(True),
        #
        # Are we subtracting pileup as 4-vector?
        subtractPileupAs4Vec = cms.bool(False)
    )
    return config

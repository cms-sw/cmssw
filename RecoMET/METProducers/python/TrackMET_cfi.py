import FWCore.ParameterSet.Config as cms

# File: TrackMET.cff
# Original Author: C. Veelken
# Date: 05.02.2013
#
# Form uncorrected Missing ET from CKF tracks and store into event as a MET
# product

trackMet = cms.EDProducer("TrackMETProducer",
    src = cms.InputTag("generalTracks"),
    globalThreshold = cms.double(0.)
)

import FWCore.ParameterSet.Config as cms

# File: HTMET.cff
# Author: R. Cavanaugh
# Date: 08.08.2006
#
# Form uncorrected Missing HT from Jets and store into event as a MET
# product
htMet = cms.EDProducer("METProducer",
    src = cms.InputTag("midPointCone5CaloJets"),
    METType = cms.string('MET'),
    alias = cms.string('HTMET'),
    noHF = cms.bool(False),
    globalThreshold = cms.double(20.0),
    InputType = cms.string('CaloJetCollection')
)

htMetSC5 = cms.EDProducer("METProducer",
    src = cms.InputTag("sisCone5CaloJets"),
    METType = cms.string('MET'),
    alias = cms.string('HTMETSC5'),
    noHF = cms.bool(False),
    globalThreshold = cms.double(20.0),
    InputType = cms.string('CaloJetCollection')
)

htMetSC7 = cms.EDProducer("METProducer",
    src = cms.InputTag("sisCone7CaloJets"),
    METType = cms.string('MET'),
    alias = cms.string('HTMETSC7'),
    noHF = cms.bool(False),
    globalThreshold = cms.double(20.0),
    InputType = cms.string('CaloJetCollection')
)

htMetIC5 = cms.EDProducer("METProducer",
    src = cms.InputTag("iterativeCone5CaloJets"),
    METType = cms.string('MET'),
    alias = cms.string('HTMETIC5'),
    noHF = cms.bool(False),
    globalThreshold = cms.double(20.0),
    InputType = cms.string('CaloJetCollection')
)

htMetKT4 = cms.EDProducer("METProducer",
    src = cms.InputTag("kt4CaloJets"),
    METType = cms.string('MET'),
    alias = cms.string('HTMETKT4'),
    noHF = cms.bool(False),
    globalThreshold = cms.double(20.0),
    InputType = cms.string('CaloJetCollection')
)

htMetKT6 = cms.EDProducer("METProducer",
    src = cms.InputTag("kt6CaloJets"),
    METType = cms.string('MET'),
    alias = cms.string('HTMETKT6'),
    noHF = cms.bool(False),
    globalThreshold = cms.double(20.0),
    InputType = cms.string('CaloJetCollection')
)

htMetAK5 = cms.EDProducer("METProducer",
    src = cms.InputTag("ak5CaloJets"),
    METType = cms.string('MET'),
    alias = cms.string('HTMETAK5'),
    noHF = cms.bool(False),
    globalThreshold = cms.double(20.0),
    InputType = cms.string('CaloJetCollection')
)

htMetAK7 = cms.EDProducer("METProducer",
    src = cms.InputTag("ak7CaloJets"),
    METType = cms.string('MET'),
    alias = cms.string('HTMETAK7'),
    noHF = cms.bool(False),
    globalThreshold = cms.double(20.0),
    InputType = cms.string('CaloJetCollection')
)


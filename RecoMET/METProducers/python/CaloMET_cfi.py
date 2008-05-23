import FWCore.ParameterSet.Config as cms

# File: CaloMET.cff
# Author: R. Cavanaugh
# Date: 08.08.2006
#
# Form uncorrected Missing ET from Calorimeter Towers and store into event as a CaloMET
# product
met = cms.EDProducer("METProducer",
    src = cms.InputTag("towerMaker"),
    METType = cms.string('CaloMET'),
    alias = cms.string('RawCaloMET'),
    noHF = cms.bool(False),
    globalThreshold = cms.double(0.5),
    InputType = cms.string('CandidateCollection')
)

metOpt = cms.EDProducer("METProducer",
    src = cms.InputTag("calotoweroptmaker"),
    METType = cms.string('CaloMET'),
    alias = cms.string('RawCaloMETOpt'),
    noHF = cms.bool(False),
    globalThreshold = cms.double(0.0),
    InputType = cms.string('CandidateCollection')
)

metNoHF = cms.EDProducer("METProducer",
    src = cms.InputTag("towerMaker"),
    METType = cms.string('CaloMET'),
    alias = cms.string('RawCaloMETNoHF'),
    noHF = cms.bool(True),
    globalThreshold = cms.double(0.5),
    InputType = cms.string('CandidateCollection')
)

metOptNoHF = cms.EDProducer("METProducer",
    src = cms.InputTag("calotoweroptmaker"),
    METType = cms.string('CaloMET'),
    alias = cms.string('RawCaloMETOptNoHF'),
    noHF = cms.bool(True),
    globalThreshold = cms.double(0.0),
    InputType = cms.string('CandidateCollection')
)



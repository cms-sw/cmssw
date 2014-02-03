import FWCore.ParameterSet.Config as cms

# File: CaloMET.cff
# Original Author: R. Cavanaugh
# Date: 08.08.2006
#
# Form uncorrected Missing ET from Calorimeter Towers and store into event as a CaloMET
# product

# Modification by F. Ratnikov and R. Remington
# Date: 10/21/08
# Additional modules available for MET Reconstruction using towers w/wo HO included



from RecoMET.METProducers.METSigParams_cfi import *

met = cms.EDProducer(
    "METProducer",
    METSignificance_params,
    src = cms.InputTag("towerMaker"),
    METType = cms.string('CaloMET'),
    alias = cms.string('RawCaloMET'),
    noHF = cms.bool(False),
    globalThreshold = cms.double(0.3),
    InputType = cms.string('CandidateCollection'),
    calculateSignificance = cms.bool(True)
    )

metHO = met.clone()
metHO.src = "towerMakerWithHO"
metHO.alias = 'RawCaloMETHO'

metOpt = cms.EDProducer(
    "METProducer",
    METSignificance_params,
    src = cms.InputTag("calotoweroptmaker"),
    METType = cms.string('CaloMET'),
    alias = cms.string('RawCaloMETOpt'),
    noHF = cms.bool(False),
    globalThreshold = cms.double(0.0),
    InputType = cms.string('CandidateCollection'),
    calculateSignificance = cms.bool(True)
    )

metOptHO = metOpt.clone()
metOptHO.src = "calotoweroptmakerWithHO"
metOptHO.alias = 'RawCaloMETOptHO'

metNoHF = cms.EDProducer(
    "METProducer",
    METSignificance_params,
    src = cms.InputTag("towerMaker"),
    METType = cms.string('CaloMET'),
    alias = cms.string('RawCaloMETNoHF'),
    noHF = cms.bool(True),
    globalThreshold = cms.double(0.3),
    InputType = cms.string('CandidateCollection'),
    calculateSignificance = cms.bool(True)
)

metNoHFHO = metNoHF.clone()
metNoHFHO.src = "towerMakerWithHO"
metNoHFHO.alias = 'RawCaloMETNoHFHO'

metOptNoHF = cms.EDProducer(
    "METProducer",
    METSignificance_params,
    src = cms.InputTag("calotoweroptmaker"),
    METType = cms.string('CaloMET'),
    alias = cms.string('RawCaloMETOptNoHF'),
    noHF = cms.bool(True),
    globalThreshold = cms.double(0.0),
    InputType = cms.string('CandidateCollection'),
    calculateSignificance = cms.bool(True)
    )
metOptNoHFHO = metOptNoHF.clone()
metOptNoHFHO.src = "calotoweroptmakerWithHO"
metOptNoHFHO.alias = 'RawCaloMETOptNoHFHO'



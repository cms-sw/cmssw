import FWCore.ParameterSet.Config as cms

hltESPPhase2StripCPE = cms.ESProducer("Phase2StripCPEESProducer",
    ComponentType = cms.string('Phase2StripCPE'),
    appendToDataLabel = cms.string('HLT'),
    parameters = cms.PSet(
        LorentzAngle_DB = cms.bool(True),
        TanLorentzAnglePerTesla = cms.double(0.07)
    )
)

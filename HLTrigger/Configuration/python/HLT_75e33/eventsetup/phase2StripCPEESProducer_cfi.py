import FWCore.ParameterSet.Config as cms

phase2StripCPEESProducer = cms.ESProducer("Phase2StripCPEESProducer",
    ComponentType = cms.string('Phase2StripCPE'),
    parameters = cms.PSet(
        LorentzAngle_DB = cms.bool(False),
        TanLorentzAnglePerTesla = cms.double(0.07)
    )
)

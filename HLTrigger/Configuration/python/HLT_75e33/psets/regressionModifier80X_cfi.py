import FWCore.ParameterSet.Config as cms

regressionModifier80X = cms.PSet(
    applyExtraHighEnergyProtection = cms.bool(True),
    autoDetectBunchSpacing = cms.bool(True),
    bunchSpacingTag = cms.InputTag("bunchSpacingProducer"),
    electron_config = cms.PSet(
        combinationKey_25ns = cms.string('gedelectron_p4combination_25ns'),
        combinationKey_50ns = cms.string('gedelectron_p4combination_50ns'),
        regressionKey_25ns = cms.vstring(
            'gedelectron_EBCorrection_25ns',
            'gedelectron_EECorrection_25ns'
        ),
        regressionKey_50ns = cms.vstring(
            'gedelectron_EBCorrection_50ns',
            'gedelectron_EECorrection_50ns'
        ),
        uncertaintyKey_25ns = cms.vstring(
            'gedelectron_EBUncertainty_25ns',
            'gedelectron_EEUncertainty_25ns'
        ),
        uncertaintyKey_50ns = cms.vstring(
            'gedelectron_EBUncertainty_50ns',
            'gedelectron_EEUncertainty_50ns'
        )
    ),
    manualBunchSpacing = cms.int32(50),
    modifierName = cms.string('EGRegressionModifierV1'),
    photon_config = cms.PSet(
        regressionKey_25ns = cms.vstring(
            'gedphoton_EBCorrection_25ns',
            'gedphoton_EECorrection_25ns'
        ),
        regressionKey_50ns = cms.vstring(
            'gedphoton_EBCorrection_50ns',
            'gedphoton_EECorrection_50ns'
        ),
        uncertaintyKey_25ns = cms.vstring(
            'gedphoton_EBUncertainty_25ns',
            'gedphoton_EEUncertainty_25ns'
        ),
        uncertaintyKey_50ns = cms.vstring(
            'gedphoton_EBUncertainty_50ns',
            'gedphoton_EEUncertainty_50ns'
        )
    ),
    rhoCollection = cms.InputTag("fixedGridRhoFastjetAll"),
    vertexCollection = cms.InputTag("offlinePrimaryVertices")
)
import FWCore.ParameterSet.Config as cms

ecalDetIdToBeRecovered = cms.EDProducer("EcalDetIdToBeRecoveredProducer",

        # SRP collections
        ebSrFlagCollection = cms.InputTag("ecalDigis"),
        eeSrFlagCollection = cms.InputTag("ecalDigis"),

        # Integrity for xtal data
        ebIntegrityGainErrors = cms.InputTag("ecalDigis:EcalIntegrityGainErrors"),
        ebIntegrityGainSwitchErrors = cms.InputTag("ecalDigis:EcalIntegrityGainSwitchErrors"),
        ebIntegrityChIdErrors = cms.InputTag("ecalDigis:EcalIntegrityChIdErrors"),

        # Integrity for xtal data - EE specific (to be rivisited towards EB+EE common collection)
        eeIntegrityGainErrors = cms.InputTag("ecalDigis:EcalIntegrityGainErrors"),
        eeIntegrityGainSwitchErrors = cms.InputTag("ecalDigis:EcalIntegrityGainSwitchErrors"),
        eeIntegrityChIdErrors = cms.InputTag("ecalDigis:EcalIntegrityChIdErrors"),

        # Integrity Errors
        integrityTTIdErrors = cms.InputTag("ecalDigis:EcalIntegrityTTIdErrors"),
        integrityBlockSizeErrors = cms.InputTag("ecalDigis:EcalIntegrityBlockSizeErrors"),

        # output collections
        ebDetIdToBeRecovered = cms.string("ebDetId"),
        eeDetIdToBeRecovered = cms.string("eeDetId"),
        ebFEToBeRecovered = cms.string("ebFE"),
        eeFEToBeRecovered = cms.string("eeFE")
)

import FWCore.ParameterSet.Config as cms

modMTCCAmplifyDigis = cms.EDFilter("MTCCAmplifyDigis",
    # Sigmas for different Subdetectors
    oDigiAmplifySigmas = cms.untracked.PSet(
        dTOB = cms.untracked.double(5.0),
        dTIB = cms.untracked.double(4.0)
    ),
    oNewSiStripDigisLabel = cms.untracked.string('SiStripAmplifiedDigis'),
    # Scale factors ParameterSet
    oDigiScaleFactors = cms.untracked.PSet(
        oTIB = cms.untracked.PSet(
            dL2 = cms.untracked.double(0.914),
            dL1 = cms.untracked.double(0.857)
        ),
        oTOB = cms.untracked.PSet(
            dL2 = cms.untracked.double(0.854),
            dL1 = cms.untracked.double(0.935)
        )
    ),
    oSiStripDigisProdInstName = cms.untracked.string(''),
    oSiStripDigisLabel = cms.untracked.string('siStripDigis')
)



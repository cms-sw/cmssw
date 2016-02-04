import FWCore.ParameterSet.Config as cms

SectCollParametersBlock = cms.PSet(
    SectCollParameters = cms.PSet(
        # Programmable Coarse Sync BX adjustment for SC in MB4 (double station)
        SCCSP5 = cms.int32(0),
        # Programmable Coarse Sync BX adjustment for SC in MB2
        SCCSP2 = cms.int32(0),
        # Programmable Coarse Sync BX adjustment for SC in MB3
        SCCSP3 = cms.int32(0),
        # Enabling carry in SC station MB4
        SCECF4 = cms.bool(False),
        # Programmable Coarse Sync BX adjustment for SC in MB1
        SCCSP1 = cms.int32(0),
        # Enabling carry in SC station MB2
        SCECF2 = cms.bool(False),
        # Enabling carry in SC station MB3
        SCECF3 = cms.bool(False),
        # Programmable Coarse Sync BX adjustment for SC in MB4
        SCCSP4 = cms.int32(0),
        # Enabling carry in SC station MB1
        SCECF1 = cms.bool(False),
        # Debug falg
        Debug = cms.untracked.bool(False)
    )
)



import FWCore.ParameterSet.Config as cms

SiStripConfObjectGenerator = cms.Service(
    "SiStripConfObjectGenerator",
    Parameters = cms.VPSet(
        cms.PSet(
            ParameterName = cms.string("par1"),
            ParameterValue = cms.int32(1),
        ),
        cms.PSet(
            ParameterName = cms.string("par2"),
            ParameterValue = cms.int32(2),
        ),
    )
)
